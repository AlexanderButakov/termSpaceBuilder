# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import os, sys
from os.path import join
from collections import defaultdict
from nltk.stem.snowball import RussianStemmer
from nltk.stem.snowball import GermanStemmer
from porter import PorterStemmer
import codecs, re, json
import pymorphy2


# Скрипт для составления словаря основ слов с частотой их встречаемости
# по документам корпусов. В зависимости от переданного параметра 
# выходной json можно использовать: 1) tfidf - для реализации подсчета
# tf-idf, 2) raw - для подсчёта абсолютной частоты основы по всем документам.
# Словарь содержит основу и указатель в скольки документах
# корпуса основа встретилась (или абсолютную частоту). 
# {"football":2} означает, что основа football встретилась в двух документах из всего корпуса.
# 
# Скрипт для трёх языков: английского, немецкого, русского.
# Класс LoadExternalLists загружает вспомогательные файлы,
# класс NormalizerDE - лемматизация немецких текстов,
# класс NormalizerRU - только функция преобразования ё в е,
# класс NormalizerEN - функции обработки слова на английском.
# Для работы требует наличие модуля Porter Stemmer,
# pymorphy2 и nltk.


class LoadExternalLists(object):
    
    """
    Загружаем в память файл стоп-слов, файл с неправ. формами глаголов 
    и файл с неправ. формами множ. числа сущ-х.
    """

    def __init__(self):

        self.stopwords_en = set()
        self.stopwords_de = set()
        self.stopwords_ru = set()
        
        self.verbtransforms = {}
        self.nountransforms = {}

        self.lexicon_de = {}
        
        
    def loadStopWordsEN(self):

        with codecs.open(r".\txt_resources\stopwords_en.txt",'r','utf-16') as file_openstopw:
        	self.stopwords = set(file_openstopw.read().split('\r\n'))

        return self.stopwords


    def loadStopWordsDE(self):

        with codecs.open(r".\txt_resources\stopwords_de.txt",'r','utf-16') as file_openstopw:
        	self.stopwords = set(file_openstopw.read().split('\r\n'))

        return self.stopwords


    def loadStopWordsRU(self):

        with codecs.open(r".\txt_resources\stopwords_ru.txt",'r','utf-16') as file_openstopw:
        	self.stopwords = set(file_openstopw.read().split('\r\n'))

        return self.stopwords


    def loadVerbForms(self):

        with codecs.open(r'.\txt_resources\verbforms.txt','r','utf-16') as verbforms:
	        for line in verbforms:
	            line_part = line.strip().split('\t')
	            if line_part[0] not in self.verbtransforms:
	                self.verbtransforms[line_part[0]] = line_part[1]

        return self.verbtransforms

    def loadNounforms(self):

        with codecs.open(r'.\txt_resources\nounforms.txt','r','utf-16') as nounforms:
	        for line in nounforms:
	            line_part = line.strip().split('\t')
	            if line_part[0] not in self.nountransforms:
	                self.nountransforms[line_part[0]] = line_part[1]

        return self.nountransforms

    def loadLexiconDe(self):

        with open(r'.\lexicon\lexicon_dict_49289.json', 'r') as infile:

            self.lexicon_de = json.load(infile)

        return self.lexicon_de


class NormalizerDE(object):

	
	def normalizeUmlaut(self, word):
		"""
		Конвертируем немекцие умлауты для унификации. Обратного преобразования не предусмотрено.
		"""

		# cyr_err = {'д':'ä', 'ц':'ö', 'ь':'ü', 'Я':'ß', 'Д':'Ä', 'Ц':'Ö', 'Ь':'Ü'}
		# umlauts_case = {'ä':'ae', 'ö':'oe', 'ü':'ue', 'ß':'ss', 'Ä':'Ae', 'Ö':'Oe', 'Ü':'ue', 'д':'ae', 'ц':'oe', 'ь':'ue', 'Я':'ss', 'Д':'Ae', 'Ц':'Oe', 'Ь':'Ue'}
		umlauts = {'ä':'ae', 'ö':'oe', 'ü':'ue', 'ß':'ss', 'д':'ae', 'ц':'oe', 'ь':'ue', 'Я':'ss', 'ä':'ae', 'ö':'oe', 'ü':'ue'}

		for umlaut, ersatz in umlauts.iteritems():
			if umlaut in word:
				word = word.replace(umlaut, ersatz)
				
		return word


	def deleteContrs(self, str1):
		"""
		Избавляемся от сокращений в конце слов путем метода re.sub (заменяем окончание на ничто).
		Если паттерн окончания всё ещё найден у слова (англ: I'd've), то ещё раз применяем метод re.sub.
		"""

		# для удаления в конце слов сокращений типа you've, don't и пр.
		del_endings = re.compile(r'[\'\’`‘]+[s|m|t|d|n]$|[\'\’‘`]+(ve|ll|re|nt|ya|yer)$')

		no_endings = del_endings.sub('', str1)

		if del_endings.search(no_endings):

			no_endings2 = del_endings.sub('', no_endings)
			
			return no_endings2
		else:

			return no_endings


	def lemmatize(self, word, lexicon):
		"""
		Функция лемматизации немецких слов.
		Подгружается особым образом сформированный лексикон.
		Структура лексикона: питоновские вложенные словари.
		Первый уровень - первые три буквы слова (см. перем. alphabet).
		Внутри первого уровня слова, начинающиеся на эту букву, ключ - лемма,
		значение - список словоформ этой леммы. Выделение трех букв сделано
		для того, чтобы как-то уменьшить время поиска словоформы. Так не надо
		пробегать по всему словарю, а искать только в одном подсловаре.
		Поиск:
		Поиск идёт по списку словоформ, пока не встретится подходящая форма.
		Сначала слово ищется целиком, если не находится, откусывается первая
		буква слева и так до конца. Таким образом обрабатываются новые 
		слова с префиксами (ueberbakterien) и слова-компаунды (krankenversicherungsbescheinigungen),
		которых нет в ktrcbrjyt, но есть их последняя часть (bescheinigungen).
		"""

		alphabet = tuple(["aac", "aal", "aas", "aba", "abb", "abc", "abd", "abe", "abf", "abg", "abh", "abi", "abj", "abk", "abl", "abm", "abn", "abo", "abp", "abq", "abr", "abs", "abt", "abu", "abv", "abw", "abz", "acc", "ach", "ack", "act", "ada", "add", "ade", "adh", "adj", "adl", "adm", "ado", "adr", "ads", "adv", "aeb", "aec", "aed", "aef", "aeg", "aeh", "ael", "aem", "aen", "aep", "aeq", "aer", "aes", "aet", "aeu", "aex", "aff", "afg", "afr", "aft", "aga", "age", "agg", "agi", "ago", "agr", "ags", "ahl", "ahm", "ahn", "aho", "air", "aka", "akk", "akn", "ako", "akq", "akr", "akt", "aku", "akw", "akz", "ala", "alb", "alc", "ale", "alf", "alg", "ali", "alk", "all", "alm", "alp", "alr", "als", "alt", "alu", "alz", "ama", "amb", "ame", "amh", "ami", "amm", "amn", "amo", "amp", "ams", "amt", "amu", "ana", "anb", "and", "ane", "anf", "ang", "anh", "ani", "anj", "ank", "anl", "anm", "ann", "ano", "anp", "anq", "anr", "ans", "ant", "anv", "anw", "anz", "aor", "apa", "ape", "apf", "aph", "apo", "app", "apr", "aps", "aqu", "ara", "arb", "arc", "ard", "are", "arg", "arh", "ari", "ark", "arm", "arn", "aro", "arr", "ars", "art", "arz", "asb", "asc", "ase", "asi", "ask", "aso", "asp", "ass", "ast", "asy", "asz", "ate", "ath", "atl", "atm", "ato", "atr", "att", "atu", "aub", "aud", "aue", "auf", "aug", "auh", "auk", "aul", "aup", "aur", "aus", "aut", "ava", "ave", "avi", "axe", "axi", "aza", "aze", "azt", "azu", "bab", "bac", "bad", "bae", "baf", "bag", "bah", "bai", "baj", "bak", "bal", "bam", "ban", "bap", "bar", "bas", "bat", "bau", "bay", "baz", "bea", "beb", "bec", "bed", "bee", "bef", "beg", "beh", "bei", "bej", "bek", "bel", "bem", "ben", "beo", "bep", "beq", "ber", "bes", "bet", "beu", "bev", "bew", "bez", "bia", "bib", "bie", "big", "bik", "bil", "bim", "bin", "bio", "bir", "bis", "bit", "biw", "biz", "bjo", "bla", "ble", "bli", "blo", "blu", "bmw", "boa", "bob", "boc", "bod", "boe", "bog", "boh", "boi", "boj", "bol", "bom", "bon", "boo", "bor", "bos", "bot", "bou", "bow", "box", "boy", "bra", "bre", "bri", "bro", "bru", "bub", "buc", "bud", "bue", "buf", "bug", "buh", "buk", "bul", "bum", "bun", "bur", "bus", "but", "byt", "byz", "cac", "cad", "cae", "caf", "cag", "cal", "cam", "can", "cap", "car", "cas", "cd-", "cds", "cea", "cel", "cem", "cen", "ceo", "cer", "ces", "cey", "cha", "che", "chi", "chl", "cho", "chr", "cie", "cin", "cit", "cla", "cle", "cli", "clo", "clu", "coa", "coc", "cod", "coh", "coi", "cok", "col", "com", "con", "coo", "cop", "cor", "cou", "cov", "cow", "coy", "cpu", "cra", "cre", "cro", "cto", "cup", "cur", "cut", "cya", "cyb", "cyr", "dab", "dac", "dad", "dae", "daf", "dag", "dah", "dai", "dak", "dal", "dam", "dan", "dar", "das", "dat", "dau", "dav", "daw", "daz", "dea", "deb", "dec", "ded", "dee", "def", "deg", "deh", "dei", "dej", "dek", "del", "dem", "den", "deo", "dep", "der", "des", "det", "deu", "dev", "dez", "di.", "dia", "dic", "did", "die", "dif", "dig", "dik", "dil", "dim", "din", "dio", "dip", "dir", "dis", "dit", "div", "diw", "djs", "do.", "doc", "dod", "doe", "dog", "doh", "dok", "dol", "dom", "don", "doo", "dop", "dor", "dos", "dot", "dou", "dow", "doz", "dra", "dre", "dri", "dro", "dru", "dsc", "dtu", "dua", "dub", "duc", "dud", "due", "duf", "dui", "dul", "dum", "dun", "duo", "dup", "dur", "dus", "dut", "duz", "dvd", "dyn", "dys", "ebb", "ebe", "ebn", "ech", "eck", "ecu", "edb", "edd", "ede", "edg", "edi", "edl", "edm", "edu", "efe", "eff", "ega", "ege", "egg", "ego", "ehe", "ehr", "eib", "eic", "eid", "eie", "eif", "eig", "eil", "eim", "ein", "eis", "eit", "eiw", "eiz", "eja", "eje", "eke", "ekl", "eks", "ekz", "ela", "elc", "eld", "ele", "elf", "eli", "elk", "ell", "elm", "elo", "els", "elt", "ema", "emb", "eme", "emi", "emm", "emo", "emp", "ems", "emu", "enc", "end", "ene", "enf", "eng", "enk", "eno", "ens", "ent", "enu", "enz", "epe", "epi", "epo", "epp", "eps", "equ", "era", "erb", "erc", "erd", "ere", "erf", "erg", "erh", "eri", "erj", "erk", "erl", "erm", "ern", "ero", "erp", "erq", "err", "ers", "ert", "eru", "erw", "erz", "esc", "ese", "esk", "eso", "esp", "ess", "est", "eta", "eth", "eti", "etu", "etw", "ety", "euc", "eug", "eul", "eun", "eup", "eur", "eut", "eva", "eve", "evi", "evo", "ewi", "exa", "exc", "exe", "exh", "exi", "exk", "exm", "exn", "exo", "exp", "exq", "exs", "ext", "exz", "eyl", "fab", "fac", "fad", "fae", "fag", "fah", "fai", "fak", "fal", "fam", "fan", "far", "fas", "fat", "fau", "fav", "fax", "faz", "fea", "feb", "fec", "fed", "fee", "feg", "feh", "fei", "fel", "fem", "fen", "fer", "fes", "fet", "feu", "fez", "fia", "fib", "fic", "fid", "fie", "fig", "fik", "fil", "fim", "fin", "fir", "fis", "fit", "fix", "fjo", "fla", "fle", "fli", "flo", "flu", "fly", "foc", "foe", "foh", "fok", "fol", "fon", "fop", "for", "fos", "fot", "fou", "foy", "fr.", "fra", "fre", "fri", "fro", "fru", "fuc", "fue", "fuf", "fug", "fuh", "ful", "fum", "fun", "fur", "fus", "fut", "g'f", "g'm", "g'n", "g'r", "g's", "g'w", "gab", "gad", "gae", "gaf", "gag", "gal", "gam", "gan", "gar", "gas", "gat", "gau", "gaz", "gbs", "gby", "gea", "geb", "gec", "ged", "gee", "gef", "geg", "geh", "gei", "gej", "gek", "gel", "gem", "gen", "geo", "gep", "geq", "ger", "ges", "get", "geu", "gev", "gew", "gez", "gha", "ghe", "ghu", "gib", "gie", "gif", "gig", "gil", "gim", "gin", "gip", "gir", "gis", "git", "giu", "gla", "gle", "gli", "glo", "glu", "gly", "gmb", "gna", "gne", "gno", "gnu", "gob", "goc", "god", "goe", "gog", "gol", "gom", "gon", "gor", "gos", "got", "gou", "gra", "gre", "gri", "gro", "gru", "gua", "guc", "gud", "gue", "gui", "gul", "gum", "gun", "gur", "gus", "gut", "gym", "gyn", "gys", "g‘f", "g‘m", "g‘n", "g‘r", "g‘s", "g‘w", "g’f", "g’m", "g’n", "g’r", "g’s", "g’w", "haa", "hab", "hac", "had", "hae", "haf", "hag", "hah", "hai", "hak", "hal", "ham", "han", "hap", "har", "has", "hat", "hau", "hav", "haw", "hax", "hay", "haz", "hea", "heb", "hec", "hed", "hee", "hef", "heg", "heh", "hei", "hek", "hel", "hem", "hen", "her", "hes", "het", "heu", "hex", "hey", "hic", "hie", "hig", "hil", "him", "hin", "hip", "hir", "his", "hit", "hiw", "hob", "hoc", "hod", "hoe", "hof", "hoh", "hok", "hol", "hom", "hon", "hoo", "hop", "hor", "hos", "hot", "hou", "hoy", "hub", "huc", "hue", "huf", "hug", "huh", "hul", "hum", "hun", "hup", "hur", "hus", "hut", "hya", "hyb", "hyd", "hye", "hyg", "hym", "hyp", "hys", "ibm", "ibr", "ico", "ics", "ide", "idi", "ido", "idy", "ige", "igl", "ign", "igo", "ike", "iko", "ilb", "ill", "ils", "ilt", "ima", "imb", "imi", "imk", "imm", "imp", "ina", "inb", "inc", "ind", "ine", "inf", "ing", "inh", "ini", "inj", "ink", "inl", "inn", "ino", "inq", "ins", "int", "inv", "inw", "inz", "iod", "ion", "iqs", "ira", "ird", "ire", "iri", "irl", "irm", "iro", "irr", "isa", "ise", "isl", "ism", "iso", "isr", "iss", "ist", "ita", "ite", "iva", "iza", "ize", "jac", "jae", "jag", "jah", "jak", "jal", "jam", "jan", "jap", "jar", "jas", "jau", "jaw", "jaz", "jea", "jec", "jed", "jee", "jel", "jen", "jer", "jes", "jet", "jew", "jim", "jin", "joa", "job", "joc", "jod", "joe", "jog", "joh", "joi", "jok", "jol", "jon", "jop", "jor", "jos", "jot", "jou", "jov", "jua", "jub", "juc", "jud", "jue", "jug", "jul", "jum", "jun", "jup", "jur", "jus", "jut", "juw", "jux", "jva", "kab", "kac", "kad", "kae", "kaf", "kah", "kai", "kaj", "kak", "kal", "kam", "kan", "kap", "kar", "kas", "kat", "kau", "kav", "kbi", "kbs", "kby", "keb", "kec", "kee", "kef", "keg", "keh", "kei", "kek", "kel", "kem", "ken", "ker", "kes", "ket", "keu", "kev", "kha", "kib", "kic", "kid", "kie", "kif", "kil", "kim", "kin", "kio", "kip", "kir", "kis", "kit", "kiw", "kla", "kle", "kli", "klo", "klu", "kna", "kne", "kni", "kno", "knu", "koa", "kob", "koc", "kod", "koe", "kof", "kog", "koh", "koi", "koj", "kok", "kol", "kom", "kon", "koo", "kop", "kor", "kos", "kot", "kra", "kre", "kri", "kro", "kru", "kry", "kub", "kuc", "kue", "kuf", "kug", "kuh", "kul", "kum", "kun", "kup", "kur", "kus", "kut", "kuv", "kuw", "kyb", "kyr", "kzs", "lab", "lac", "lad", "lae", "laf", "lag", "lah", "lai", "lak", "lal", "lam", "lan", "lap", "laq", "lar", "las", "lat", "lau", "lav", "law", "lax", "lay", "laz", "lea", "leb", "lec", "led", "lee", "leg", "leh", "lei", "lek", "lem", "len", "leo", "ler", "les", "let", "leu", "lev", "lex", "lia", "lib", "lic", "lid", "lie", "lif", "lig", "lii", "lik", "lil", "lim", "lin", "lip", "liq", "lis", "lit", "liv", "liz", "lkw", "lob", "loc", "lod", "loe", "lof", "log", "loh", "loi", "lok", "lol", "lom", "lon", "loo", "lor", "los", "lot", "lov", "loy", "lps", "luc", "lud", "lue", "luf", "lug", "luk", "lul", "lum", "lun", "lup", "lur", "lus", "lut", "lux", "luz", "lyk", "lym", "lyn", "lyr", "lys", "maa", "mac", "mad", "mae", "maf", "mag", "mah", "mai", "maj", "mak", "mal", "mam", "man", "map", "mar", "mas", "mat", "mau", "max", "may", "maz", "mbi", "mbs", "mby", "mec", "med", "mee", "meg", "meh", "mei", "mek", "mel", "mem", "men", "mep", "mer", "mes", "met", "meu", "mex", "mey", "mez", "mi.", "mia", "mic", "mie", "mig", "mih", "mik", "mil", "mim", "min", "mir", "mis", "mit", "mix", "mne", "mo.", "mob", "moc", "mod", "moe", "mof", "mog", "moh", "mok", "mol", "mom", "mon", "moo", "mop", "mor", "mos", "mot", "mou", "mov", "moz", "mp3", "mps", "muc", "mue", "muf", "mul", "mum", "mun", "mur", "mus", "mut", "myr", "mys", "myt", "nab", "nac", "nad", "nae", "nag", "nah", "nai", "nam", "nan", "nap", "nar", "nas", "nat", "nau", "nav", "naz", "nea", "neb", "nec", "nef", "neg", "neh", "nei", "nek", "nel", "nen", "neo", "nep", "ner", "nes", "net", "neu", "new", "nib", "nic", "nid", "nie", "nig", "nih", "nik", "nil", "nim", "nip", "nir", "nis", "nit", "niv", "nix", "niz", "nob", "noc", "noe", "nom", "non", "nop", "nor", "nos", "not", "nou", "nov", "now", "nua", "nud", "nue", "nuk", "nul", "num", "nus", "nut", "nva", "nyl", "nym", "oas", "obd", "obe", "obg", "obi", "obj", "obl", "obm", "obo", "obr", "obs", "obt", "obu", "obz", "och", "ock", "ode", "odi", "odo", "ody", "oed", "oef", "oeh", "oek", "oel", "oen", "oer", "oes", "ofe", "off", "ohn", "ohr", "okk", "okt", "oku", "okz", "ola", "old", "ole", "olf", "olg", "oli", "oll", "oly", "oma", "ome", "omi", "omn", "ona", "onk", "ont", "ony", "opa", "ope", "opf", "opi", "opo", "opp", "opt", "opu", "ora", "orb", "orc", "ord", "org", "ori", "ork", "orl", "orn", "ort", "osc", "osk", "osl", "osm", "osn", "osr", "oss", "ost", "osz", "oto", "ott", "out", "ouv", "ouz", "ova", "ove", "ovu", "owe", "oxi", "oxy", "oze", "ozo", "paa", "pac", "pad", "pae", "paf", "pag", "pak", "pal", "pam", "pan", "pap", "par", "pas", "pat", "pau", "pav", "paz", "pcs", "pda", "pec", "ped", "peg", "pei", "pej", "pek", "pel", "pen", "pep", "per", "pes", "pet", "pfa", "pfe", "pfi", "pfl", "pfo", "pfr", "pfu", "pha", "phe", "phi", "phl", "pho", "phr", "phy", "pia", "pic", "pie", "pig", "pik", "pil", "pim", "pin", "pio", "pip", "pir", "pis", "pit", "pix", "piz", "pkw", "pla", "ple", "pli", "plo", "plu", "pne", "poc", "pod", "poe", "pog", "poh", "poi", "pok", "pol", "pom", "pon", "poo", "pop", "por", "pos", "pot", "pow", "pra", "pre", "pri", "pro", "prs", "pru", "psa", "psc", "pse", "psy", "pub", "puc", "pud", "pue", "puf", "pul", "pum", "pun", "pup", "pur", "pus", "put", "puz", "pvc", "pyg", "pyj", "pyr", "pyt", "qua", "que", "qui", "quo", "rab", "rac", "rad", "rae", "raf", "rag", "rah", "rai", "rak", "ral", "ram", "ran", "rap", "rar", "ras", "rat", "rau", "rav", "raz", "rea", "reb", "rec", "red", "ree", "ref", "reg", "reh", "rei", "rej", "rek", "rel", "rem", "ren", "reo", "rep", "req", "res", "ret", "reu", "rev", "rex", "rez", "rha", "rhe", "rhi", "rho", "rhy", "rib", "ric", "rie", "rif", "rig", "rik", "ril", "rin", "rio", "rip", "ris", "rit", "riv", "riz", "roa", "rob", "roc", "rod", "roe", "rog", "roh", "rok", "rol", "rom", "ron", "ros", "rot", "rou", "row", "roy", "rtl", "rua", "rub", "ruc", "rud", "rue", "ruf", "ruh", "rui", "rum", "run", "rup", "rus", "rut", "rws", "ryb", "sa.", "saa", "sab", "sac", "sad", "sae", "saf", "sag", "sah", "sai", "sak", "sal", "sam", "san", "sap", "sar", "sas", "sat", "sau", "sav", "sax", "sca", "sce", "sch", "sci", "scr", "sea", "seb", "sec", "sed", "see", "seg", "seh", "sei", "sek", "sel", "sem", "sen", "seo", "sep", "seq", "ser", "ses", "set", "seu", "sex", "sez", "sha", "she", "shi", "sho", "shr", "shu", "sia", "sib", "sic", "sid", "sie", "sig", "sik", "sil", "sim", "sin", "sip", "sir", "sis", "sit", "ska", "ske", "ski", "skl", "sko", "skr", "sku", "sky", "sla", "sli", "slo", "slu", "sma", "smo", "sna", "sni", "sno", "so.", "soa", "soc", "sod", "soe", "sof", "sog", "soh", "soj", "sol", "som", "son", "sop", "sor", "sos", "sot", "sou", "sow", "soz", "spa", "spe", "sph", "spi", "spl", "spo", "spr", "spu", "squ", "sta", "ste", "sti", "sto", "str", "stu", "sty", "sub", "suc", "sud", "sue", "suf", "sug", "suh", "sui", "suj", "suk", "sul", "sum", "sup", "sur", "sus", "sut", "sve", "swe", "swi", "syl", "sym", "syn", "syr", "sys", "sze", "tab", "tac", "tad", "tae", "taf", "tag", "tai", "tak", "tal", "tam", "tan", "tao", "tap", "tar", "tas", "tat", "tau", "tav", "tax", "tay", "tby", "tea", "tec", "ted", "tee", "teg", "teh", "tei", "tek", "tel", "tem", "ten", "tep", "ter", "tes", "teu", "tex", "tha", "the", "thi", "tho", "thr", "thu", "thy", "tib", "tic", "tid", "tie", "tig", "til", "tim", "tin", "tip", "tir", "tis", "tit", "toa", "tob", "toc", "tod", "toe", "toh", "toi", "tok", "tol", "tom", "ton", "too", "top", "tor", "tos", "tot", "tou", "tox", "toy", "tra", "tre", "tri", "tro", "tru", "tsa", "tsc", "tse", "tub", "tuc", "tue", "tug", "tul", "tum", "tun", "tup", "tur", "tus", "tut", "tvs", "twi", "tyc", "typ", "tyr", "udo", "ueb", "uel", "uep", "uer", "ufe", "uff", "ufr", "uhr", "uhu", "ukr", "uku", "ulf", "uli", "ulk", "ulm", "ulr", "ult", "uma", "umb", "umc", "umd", "ume", "umf", "umg", "umh", "umi", "umj", "umk", "uml", "umm", "umn", "umo", "ump", "umq", "umr", "ums", "umt", "umv", "umw", "umz", "una", "unb", "unc", "und", "une", "unf", "ung", "unh", "uni", "unk", "unl", "unm", "unn", "uno", "unp", "unq", "unr", "uns", "unt", "unu", "unv", "unw", "unz", "upd", "upg", "ura", "urb", "urd", "ure", "urf", "urg", "urh", "uri", "urk", "url", "urm", "urn", "uro", "urp", "urs", "urt", "uru", "urv", "urw", "urz", "usa", "usb", "use", "usi", "usu", "uta", "ute", "uti", "uto", "uwe", "uze", "vae", "vag", "vak", "val", "vam", "van", "var", "vas", "vat", "veg", "veh", "vei", "vek", "ven", "ver", "ves", "vet", "via", "vib", "vic", "vid", "vie", "vik", "vil", "vio", "vip", "vir", "vis", "vit", "viz", "vla", "voe", "vog", "voi", "vok", "vol", "von", "vor", "vos", "vot", "voy", "vul", "vws", "waa", "wab", "wac", "wad", "wae", "waf", "wag", "wah", "wai", "wal", "wam", "wan", "wap", "war", "was", "wat", "way", "wcs", "web", "wec", "wed", "weg", "weh", "wei", "wel", "wen", "wer", "wes", "wet", "wgs", "whi", "wic", "wid", "wie", "wik", "wil", "wim", "win", "wip", "wir", "wis", "wit", "wla", "wms", "wob", "woc", "wod", "woe", "wog", "woh", "wok", "wol", "won", "wor", "wra", "wri", "wuc", "wue", "wul", "wun", "wup", "wur", "wus", "wut", "wyn", "x-b", "xan", "xeo", "xyl", "yac", "yan", "yeb", "yen", "yet", "yog", "yor", "yuc", "yup", "yvo", "zac", "zae", "zag", "zah", "zan", "zap", "zar", "zas", "zau", "zeb", "zec", "zed", "zeh", "zei", "zel", "zem", "zen", "zep", "zer", "zet", "zeu", "zic", "zie", "zif", "zig", "zik", "zim", "zin", "zio", "zip", "zir", "zis", "zit", "ziv", "zlo", "zob", "zoc", "zoe", "zof", "zog", "zol", "zom", "zon", "zoo", "zop", "zor", "zot", "zua", "zub", "zuc", "zud", "zue", "zuf", "zug", "zuh", "zui", "zuj", "zuk", "zul", "zum", "zun", "zuo", "zup", "zuq", "zur", "zus", "zut", "zuv", "zuw", "zuz", "zwa", "zwe", "zwi", "zwo", "zya", "zyk", "zyl", "zyn", "zyp", "zys"])

		got_lemma = False
		for l in range(len(word)):
			word_no_prefix = word[l:]
			prefix = word[:l]
			if word_no_prefix[:3] in alphabet:
				for lemma, wordforms in lexicon[word_no_prefix[:3]].iteritems():
					if word_no_prefix in tuple(wordforms):
						got_lemma = True
						return prefix+lemma

		if not got_lemma:
			return word



class NormalizerRU(object):

	
	def normalizeE(self, word):
		"""
		Конвертация русской ё --> е
		"""

		if 'ё' in word:
			word = word.replace('ё', 'е')
				
		return word



class NormalizerEN(object):


	def del_contractions(self, str1):
		"""
		Избавляемся от сокращений в конце слов путем метода re.sub (заменяем окончание на ничто).
		Если паттерн окончания всё ещё найден у слова (I'd've), то ещё раз применяем метод re.sub.
		"""

		# для удаления в конце слов сокращений типа you've, don't и пр.
		del_endings = re.compile(r'[\'\’]+[s|m|t|d|n]$|[\'\’](ve|ll|re|nt|ya|yer)$')

		no_endings = del_endings.sub('', str1)

		if del_endings.search(no_endings):

			no_endings2 = del_endings.sub('', no_endings)
			
			return no_endings2
		else:

			return no_endings

	def token_transform(self, token, irreg_verbs, irreg_nouns):
		"""
		Функция преобразовывает неправильные глаголы и форму неправильного мн.ч. сущ-х.
		Смотрит, является ли текущий токен ключом в словарях irreg_verbs и irreg_nouns,
		если является, то возвращает правильную форму, которая сидит в value словарей.
		"""

		if token in irreg_verbs:
			return irreg_verbs[token]
		elif token in irreg_nouns:
			return irreg_nouns[token]
		else:
			return token



class BuildTermSpace(object):

	"""
	Создание json-объекта в виде словаря, в котором хранятся стеммы
	значимых слов и их частотность из указанных корпусов.
	"""

	def __init__(self, language='en', action='tfidf'):

		# Вызываем LoadExternalLists, создаем список стоп-слов, 
		# загружаем немецкий лексикон,
		# 
		self.language = language
		self.action = action

		# знаки, которые будут удаляться в начале и конце токена
		self.punctuation = "∙!‼¡\"#£€$¥%&'()*+±×÷·,-./:;<=>?¿@[\]^ˆ¨_`—–­{|}~≈≠→↓¬’“”«»≫‘…¦›🌼′″¹§¼⅜½¾⅘©✒•►●★❤➡➜➚➘➔✔➓➒➑➐➏➎➍➌➋➊❸❷■†✝✌￼️³‎²‚„ ​"

		loadRes = LoadExternalLists()

		if self.language == 'de':
			self.stopwords = loadRes.loadStopWordsDE()
			# объект стеммера
			self.stemmer = GermanStemmer()
			# немецкий словарь
			print '\n', "Loading German Dictionary... OK", '\n'
			self.lexicon_de = loadRes.loadLexiconDe()
			self.normalizer = NormalizerDE()
		elif self.language == 'ru':
			self.stopwords = loadRes.loadStopWordsRU()
			self.stemmer = RussianStemmer()
			# объект pymorphy2.MorphAnalyzer(), будем использовать атрибут normal_form
			self.lemmatizer_ru = pymorphy2.MorphAnalyzer()
			self.normalizer = NormalizerRU()
		else:
			self.stopwords = loadRes.loadStopWordsEN()
			self.stemmer = PorterStemmer()
			self.normalizer = NormalizerEN()
			# список неправ. гл.
			self.irreg_verbs = loadRes.loadVerbForms()
			# список неправ. сущ-х
			self.irreg_nouns = loadRes.loadNounforms()
	

	def processString(self, line):
		"""
		Функция последовательной обработки каждого слова. Получает на вход строку, создает список tokens,
		складывает туда выделенные re.split'ом слова, 'отрезая' пунктуацию с концов слова и понижая регистр, 
		и удаляет по ходу окончания-сокращения функцией del_contractions
		Дальше переходит к формированию списка стеммированных терминов rslt_list с удалением стоп-слов
		и цифровых последовательностей.
		Возвращает список rslt_list, в котором содержатся только стеммы значимых слов.
		"""

		# для разбивки на токены по пробелам и слешам
		splitchars = re.compile(r'[\s\\\/\(\)\[\]\<\>\;\:\,\‚\—\?\!\|\"«»…#]|\.\.\.+|[ �⌂ ∞½¾►=]|\-\-|\.[\'\"’“”«»‘′″„-]') # [\.\:][\'\"’“”«»‘′″]

		# для игнорирования токенов, содержащих цифры
		esc_num = re.compile(r'[0-9]+')

		# для игнорирования URL
		#url_esc = re.compile(r'([a-z]{3,6}:\/\/)?([a-zA-Z0-9\-@?]+[\.|\:])+[a-z]{2,13}[\.\?\=\&\%\,\#\+\(\)\/\w\-]*')

				
		if self.language == 'de':
			tokens = (self.normalizer.normalizeUmlaut(self.normalizer.deleteContrs(token.strip(self.punctuation).lower())) for token in splitchars.split(line))
			rslt_list = (self.stemmer.stem(self.normalizer.lemmatize(term, self.lexicon_de)) for term in tokens if term not in self.stopwords and not esc_num.search(term) and len(term)>0)	# and not esc_num.search(term) - включить после услоия на стоп-слова, если нужно удалять токены с цифрами

		elif self.language == 'ru':
			tokens = (self.normalizer.normalizeE(token.strip(self.punctuation).lower()) for token in splitchars.split(line))
			rslt_list = (self.stemmer.stem(self.lemmatizer_ru.parse(term)[0].normal_form) for term in tokens if term not in self.stopwords and not esc_num.search(term) and len(term)>0)	# and not esc_num.search(term) - включить после услоия на стоп-слова, если нужно удалять токены с цифрами

		else:
			# генератор списка токенов: по циклу: разбиваем строку на токены по regexp splitchars,
			# 2. удаляем знаки вокруг токена, приводим к нижнему регистру, 
			# 3. трансформируем форму неправ. глаг. в правильную
			# 4. удаляем окончания-сокращения с \'
			tokens = (self.normalizer.token_transform(self.normalizer.del_contractions(token.strip(self.punctuation).lower()), self.irreg_verbs, self.irreg_nouns) for token in splitchars.split(line))

			# генератор списка терминов: если термин не в списке стоп-слов и не содержит цифр, то стеммируем его.
			rslt_list = (self.stemmer.stem(term, 0, len(term)-1) for term in tokens if term not in self.stopwords and not esc_num.search(term) and len(term)>0)	# and not esc_num.search(term) - включить после услоия на стоп-слова, если нужно удалять токены с цифрами
		

		if not rslt_list:
			return []

		else:
			return rslt_list


	def processFile(self, filename):
		"""
		Читает файл в utf-16, для каждой строки файла вызывает функцию processString,
		каждое слово из получившегося списка добавляет в set terms_set, избавляясь от
		дубликатов.
		Возвращает список уникальных лемм.
		"""

		terms_set = set()
		terms_list = []

		if self.action == 'tfidf':
			try:
				with codecs.open(filename, 'r', 'utf-16') as infile:
					
					for line in infile:
						if len(line) > 1:
							for term in self.processString(line):
								terms_set.add(term)
								
			except (UnicodeDecodeError, UnicodeError, IOError):
				pass

			return terms_set

		if self.action == 'raw':
			try:
				with codecs.open(filename, 'r', 'utf-16') as infile:
					
					for line in infile:
						if len(line) > 1:
							for term in self.processString(line):
								terms_list.append(term)
								
			except (UnicodeDecodeError, UnicodeError, IOError):
				pass

			return terms_list


	def crawl(self, dirname):
		"""
		Функция проходит по папкам и подпапкам указанной в качестве аргумента директории.
		Проверяет, если файл текстовый, то запускает функцию processFile и складывает результат
		её работы в set terms_set.
		В общем terms_dict подсчитывается частотность каждой леммы, словарь сохраняется как json.
		terms_dict отражает по сути вторую часть формулы tfidf, т.е. показывает в каком количестве
		документов встретился термин.
		"""

		docs_num = 0

		terms_dict = defaultdict(int)
		
		for root, dirs, files in os.walk(dirname):

			print root, "processing..."
			
			for filename in files:

				if filename.endswith('.txt') or filename.endswith('.TXT'):
					
					print filename

					terms_set = self.processFile(join(root,filename))

					for term in terms_set:

						terms_dict[term] += 1

					docs_num+=1
					
		if self.action == 'raw':
			with codecs.open(r'.\termSpace\\'+self.language.upper()+'frequency_list_stem.txt', 'w', 'utf-16') as outfile:
				for key, value in sorted(terms_dict.iteritems(), key=lambda x:x[1], reverse=True):
					outfile.write(key+'\t'+str(value))
					outfile.write('\n')
		
		if self.action == 'tfidf':

			with open(r".\termSpace\\" + self.language.upper() + "CorpusDict_" + str(docs_num) + ".json", 'w') as  outfile:
				json.dump(terms_dict, outfile)

		



def main():

	if len(sys.argv) != 4:
		print '\n', 'Sctipt requires parameters!','\n'
		print 'Usage: [script.py] [path_to_corpus] [en | de | ru] [tfidf | raw]'
		sys.exit(1)

	dir_path = sys.argv[1]
	language = sys.argv[2]
	# action = 1) tfidf = count stems for tfidf, 2) raw = count absolute freq. of each stem
	action = sys.argv[3]

	trms = BuildTermSpace(language, action)
	trms.crawl(dir_path)


if __name__ == '__main__':
	main()
