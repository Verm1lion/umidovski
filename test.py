import os
import re
import math
import numpy as np
import pandas as pd
import sqlalchemy as alc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def clean_text(text):
    text = str(text).replace('\n', ' ')
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tr_normalize(text):
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return str(text).translate(tr_map)

### REGEX MAVİ/BEYAZ YAKA AYRIMI ###
WHITE_OVERRIDE = re.compile(
    r'\b('
    r'mühendis\w*|muhendis\w*|engineer\w*|mimar\w*|architect\w*|'
    r'yazılımcı\w*|yazilimci\w*|developer\w*|geliştirici\w*|gelistirici\w*|programcı\w*|programci\w*|programmer\w*|coder\w*|'
    r'scrum master\w*|product owner\w*|ürün sahibi\w*|urun sahibi\w*|agile koç\w*|agile koc\w*|agile coach\w*|'
    r'veri bilimci\w*|data scientist\w*|sistem mimarı\w*|sistem mimari\w*|'
    r'tasarımcı\w*|tasarimci\w*|designer\w*|illüstratör\w*|illustrat\w*|çizer\w*|cizer\w*|'
    r'sanatçı\w*|sanatci\w*|artist\w*|modelist\w*|'
    r'fotoğrafçı\w*|fotografci\w*|photographer\w*|art director\w*|sanat yönetmeni\w*|sanat yonetmeni\w*|'
    r'görüntü yönetmeni\w*|goruntu yonetmeni\w*|supervisor\w*|yapımcı\w*|yapimci\w*|producer\w*|'
    r'yazar\w*|writer\w*|copywriter\w*|editör\w*|editor\w*|redaktör\w*|redaktor\w*|'
    r'stratejist\w*|strategist\w*|pazarlamacı\w*|pazarlamaci\w*|marketer\w*|'
    r'halkla ilişkiler\w*|halkla iliskiler\w*|'
    r'temsilci\w*|representative\w*|mümessil\w*|mumessil\w*|broker\w*|trader\w*|tüccar\w*|tuccar\w*|'
    r'distrib\w*|yatırımcı\w*|yatirimci\w*|investor\w*|'
    r'satışçı\w*|satisci\w*|satıcı\w*|satici\w*|sales\w*|'
    r'müdür\w*|mudur\w*|manager\w*|yönetici\w*|yonetici\w*|director\w*|direktör\w*|direktor\w*|'
    r'lider\w*|lead\w*|chief\w*|head\w*|başkan\w*|baskan\w*|executive\w*|koordinatör\w*|koordinator\w*|coordinator\w*|'
    r'yönetmen\w*|yonetmen\w*|'
    r'şef\b|sefi?\b|sef\w*|şefi?\b|şef\w*|'
    r'uzman\w*|specialist\w*|analist\w*|analyst\w*|danışman\w*|danisman\w*|consultant\w*|'
    r'kontrolör\w*|kontrolor\w*|controller\w*|sorumlu\w*|facilitator\w*|'
    r'sekreter\w*|asistan\w*|assistant\w*|stajyer\w*|intern\b|ofis\w*|'
    r'yetkili\w*|memur\w*|officer\w*|plancı\w*|planci\w*|planner\w*|'
    r'puantaj\w*|özlük\w*|ozluk\w*|'
    r'muhasebe\w*|mali müşavir\w*|accountant\w*|auditor\w*|denetçi\w*|denetci\w*|müfettiş\w*|mufettis\w*|ekonomist\w*|'
    r'avukat\w*|attorney\w*|lawyer\w*|müşavir\w*|musavir\w*|counsel\w*|'
    r'hrbp\w*|recruiter\w*|kurucu\w*|founder\w*|partner\w*|'
    r'öğretmen\w*|ogretmen\w*|teacher\w*|akademisyen\w*|professor\w*|instructor\w*|okutman\w*|'
    r'eğitmen\w*|egitmen\w*|trainer\w*|antrenör\w*|antrenor\w*|koç\b|koc\b|koçu\b|kocu\b|koçluk\w*|kocluk\w*|coach\w*|'
    r'doktor\w*|hekim\w*|physician\w*|cerrah\w*|surgeon\w*|diş hekimi\w*|dentist\w*|eczacı\w*|eczaci\w*|pharmacist\w*|'
    r'psikolog\w*|psychologist\w*|diyetisyen\w*|dietitian\w*|fizyoterapist\w*|physiotherapist\w*|hemşire\w*|hemsire\w*|nurse\w*|'
    r'biyolog\w*|kimyager\w*|chemist\w*|kimyacı\w*|kimyaci\w*|fizikçi\w*|fizikci\w*|istatistikçi\w*|istatistikci\w*|statistician\w*|'
    r'biyoistatistikçi\w*|biyoistatistikci\w*|araştırmacı\w*|arastirmaci\w*|researcher\w*|scientist\w*|'
    r'pilot\w*|zabit\w*|mariner\w*|kaptan pilot\w*|uzak yol kaptan\w*'
    r')\b',
    re.IGNORECASE
)

BLUE_ROLES = re.compile(
    r'\b('
    r'tersane\w*|gemici\w*|gemi adam\w*|gemi söküm\w*|gemi montaj\w*|dümenci\w*|dumenci\w*|kaptan\w*|'
    r'lostromo\w*|miço\w*|mico\w*|palamar\w*|raspa\w*|kalafat\w*|liman\w*|yağcı\w*|yagci\w*|çarkçı\w*|carkci\w*|'
    r'güverte\w*|guverte\w*|dalgıç\w*|dalgıc\w*|diver\w*|kurbağa adam\w*|kurbaga adam\w*|balıkçı\w*|balikci\w*|helmsman\w*|deckhand\w*|'
    r'boatswain\w*|stevedore\w*|dockworker\w*|shipwright\w*|'
    r'host\w*|hostes\w*|kabin memur\w*|flight attendant\w*|cabin crew\w*|yer hizmet\w*|'
    r'ramp\w*|bagaj\w*|baggage\w*|vatman\w*|makinist\w*|kondüktör\w*|konduktor\w*|de-?icing\w*|'
    r'ekskavatör\w*|eksavatör\w*|excavator\w*|bekoloder\w*|beko loder\w*|loder\w*|loader\w*|'
    r'dozer\w*|greyder\w*|grader\w*|silindir\w*|konkasör\w*|finişer\w*|finisher\w*|asfalt serici\w*|'
    r'vinç\w*|vinc\w*|crane\w*|forklift\w*|transpalet\w*|manlift\w*|bobcat\w*|sondaj\w*|sondör\w*|sondor\w*|hafriyat\w*|kazıcı\w*|kazici\w*|'
    r'kaynak\w*|welder\w*|argon\w*|tig\w*|gazaltı\w*|gazalti\w*|sualtı kaynak\w*|sualti kaynak\w*|boru montaj\w*|pipefitter\w*|'
    r'maden\w*|miner\w*|tünel\w*|tunel\w*|tbm\w*|lağımcı\w*|lagimci\w*|tahkimat\w*|ocakçı\w*|ocakci\w*|döküm\w*|dokum\w*|hadde\w*|izabe\w*|tavcı\w*|tavci\w*|'
    r'kazancı\w*|kazanci\w*|boiler\w*|lazer kesim\w*|plazma kesim\w*|abkant\w*|giyotin\w*|kumlama\w*|sandblast\w*|'
    r'galvaniz\w*|eloksal\w*|freze\w*|torna\w*|cnc\w*|pres\w*|enjeksiyon\w*|tesviye\w*|machinist\w*|millwright\w*|'
    r'endüstriyel dağcı\w*|endustriyel dagci\w*|iple erişim\w*|iple erisim\w*|rope access\w*|izolasyon\w*|yalıtım\w*|yalitim\w*|taşyünü\w*|tasyunu\w*|membran\w*|'
    r'sıva\w*|siva\w*|boya\w*|painter\w*|duvarcı\w*|duvarci\w*|mason\w*|iskele\w*|marangoz\w*|carpenter\w*|çatı\w*|cati\w*|roofer\w*|olukçu\w*|olukcu\w*|'
    r'kalıp\w*|kalip\w*|demirci\w*|beton\w*|seramik\w*|fayans\w*|mermer\w*|mozaik\w*|parke\w*|epoksi\w*|şap\w*|sap\w*|şapçı\w*|sapci\w*|'
    r'alçıpan\w*|alcipan\w*|kartonpiyer\w*|ferforje\w*|çilingir\w*|cilingir\w*|locksmith\w*|camcı\w*|camci\w*|glazier\w*|döşeme\w*|doseme\w*|upholster\w*|cıla\w*|cila\w*|lake\w*|'
    r'operatör\w*|operator\w*|usta\w*|kalfa\w*|çırak\w*|cirak\w*|işçi\w*|isci\w*|laborer\w*|worker\w*|'
    r'teknisyen\w*|tekniker\w*|technician\w*|mechanic\w*|mekaniker\w*|tamir\w*|elektrikçi\w*|elektrikci\w*|tesisatçı\w*|tesisatci\w*|plumber\w*|kombici\w*|'
    r'montaj\w*|assembl\w*|eleman\w*|personel\w*|clerk\w*|attendant\w*|yardımcı\w*|yardimci\w*|helper\w*|'
    r'güvenlik\w*|guvenlik\w*|security\w*|guard\w*|bekçi\w*|bekci\w*|koruma\w*|bodyguard\w*|itfaiye\w*|firefighter\w*|cankurtaran\w*|lifeguard\w*|'
    r'şoför\w*|sofor\w*|driver\w*|kurye\w*|courier\w*|kamyon\w*|tır\w*|tir\w*|depo\w*|yükleme\w*|yukleme\w*|boşaltma\w*|bosaltma\w*|loader\w*|unloader\w*|paketleme\w*|packer\w*|sevkiyat\w*|dağıt\w*|dagit\w*|hamal\w*|'
    r'aşçı\w*|asci\w*|cook\w*|komi\w*|bulaşık\w*|bulasik\w*|dishwasher\w*|steward\w*|fırın\w*|firin\w*|baker\w*|kasap\w*|kasab\w*|butcher\w*|kemik sıyırıcı\w*|suşi şef\w*|sushi chef\w*|'
    r'döner\w*|doner\w*|kebap\w*|pide\w*|lahmacun\w*|ızgara\w*|izgara\w*|pasta\w*|tatlı\w*|tatli\w*|börek\w*|borek\w*|baklava\w*|simit\w*|değirmen\w*|degirmen\w*|'
    r'garson\w*|waiter\w*|waitress\w*|barista\w*|barmen\w*|bartender\w*|bellboy\w*|barboy\w*|barmaid\w*|barmeid\w*|animatör\w*|animator\w*|'
    r'kat görevlisi\w*|kat hizmet\w*|housekeeper\w*|meydancı\w*|meydanci\w*|valet\w*|vale\w*|kapıcı\w*|kapici\w*|doorman\w*|vestiyer\w*|'
    r'temizlik\w*|cleaner\w*|janitor\w*|hademe\w*|çaycı\w*|cayci\w*|pompacı\w*|pompaci\w*|oto yıkama\w*|oto yikama\w*|kuru temizleme\w*|'
    r'sayacı\w*|sayaci\w*|ayakkabı\w*|ayakkabi\w*|tabakçı\w*|tabakci\w*|tanner\w*|derici\w*|saraç\w*|sarac\w*|dikiş\w*|dikis\w*|seamstress\w*|makinacı\w*|makinaci\w*|nakış\w*|nakis\w*|embroidery\w*|'
    r'overlok\w*|reçme\w*|recme\w*|ponterez\w*|ilik düğme\w*|ilik dugme\w*|ütü\w*|utu\w*|kesim\w*|cutter\w*|modelhane\w*|terzi\w*|tailor\w*|dokuma\w*|weaver\w*|ramöz\w*|ramoz\w*|apre\w*|'
    r'çiftçi\w*|ciftci\w*|ırgat\w*|irgat\w*|tarım işçisi\w*|tarim iscisi\w*|farmhand\w*|orman işçisi\w*|orman iscisi\w*|ağaç kesim\w*|agac kesim\w*|oduncu\w*|hızarcı\w*|hizarci\w*|budama\w*|aşıcı\w*|asici\w*|'
    r'çoban\w*|coban\w*|seyis\w*|nalbant\w*|sağım\w*|sagim\w*|besici\w*|arıcı\w*|arici\w*|bahçıvan\w*|bahcivan\w*|gardener\w*|seracı\w*|seraci\w*|biçerdöver\w*|bicerdover\w*|traktör\w*|traktor\w*|'
    r'hasta bakı\w*|hasta baki\w*|caregiver\w*|yaşlı bakı\w*|yasli baki\w*|engelli bakı\w*|engelli baki\w*|hasta kabul\w*|sedye\w*|portör\w*|portor\w*|orderly\w*|cna\b|'
    r'bebek bakı\w*|bebek baki\w*|bakıcı\w*|bakici\w*|'
    r'eczane kalfa\w*|anestezi teknik\w*|radyoloji teknik\w*|diş protez\w*|dis protez\w*|optisyen\w*|sterilizasyon\w*|morg\w*|gasilhane\w*|ilaçlama\w*|ilaclama\w*|haşere\w*|hasere\w*|'
    r'kasiyer\w*|cashier\w*|tezgahtar\w*|reyon\w*|'
    r'bakımcı\w*|bakimci\w*|arıza bakım\w*|ariza bakim\w*|anketör\w*|anketor\w*|'
    r'kuaför\w*|kuafor\w*|berber\w*|estetisyen\w*|'
    r'masör\w*|masor\w*|masöz\w*|masoz\w*|terapist\w*|'
    r'resepsiyonist\w*|receptionist\w*|'
    r'görevli\w*|gorevli\w*|çalışma arkadaş\w*|calisma arkadas\w*|takım üyeleri\w*|takim uyeleri\w*'
    r')\b',
    re.IGNORECASE
)

def classify_collar(text, default="WHITE"):
    if not text or not isinstance(text, str):
        return default
    clean_val = text.strip()
    if not clean_val:
        return default
    if WHITE_OVERRIDE.search(clean_val):
        return "WHITE"
    elif BLUE_ROLES.search(clean_val):
        return "BLUE"
    clean_norm = tr_normalize(clean_val)
    if WHITE_OVERRIDE.search(clean_norm):
        return "WHITE"
    elif BLUE_ROLES.search(clean_norm):
        return "BLUE"
    else:
        return default

def extract_cand_title(text):
    text_str = str(text).strip()
    match = re.search(r'Mesleki Unvan:\s*([^\n\r]+)', text_str)
    if match:
        return match.group(1).strip()
    match_ozet = re.search(r'Özet\s*/\s*Hakkında:\s*([^\n\r]+?)\s+olarak\s+\d+\s+yıllık', text_str, re.IGNORECASE)
    if match_ozet:
        return match_ozet.group(1).strip()
    for line in text_str.split('\n'):
        line_s = line.strip()
        if line_s and not line_s.lower().startswith('konum:') and not line_s.lower().startswith('özet'):
            return line_s[:100]
    return text_str.split('\n')[0][:100] if text_str else ''

# Aday Kıdem Seviyesi Çıkarımı (Unvan veya Metin Üzerinden)
def extract_cand_seniority(text_or_title):
    clean_title = extract_cand_title(text_or_title) if 'Mesleki Unvan:' in str(text_or_title) else str(text_or_title)
    clean = tr_normalize(str(clean_title)).lower()
    if re.search(r'\b(stajyer|intern|student|trainee|undergraduate|son_sinif|yeni mezun|entry|assistant|asistan|yardimci|mico|miço)\b', clean):
        return 'ENTRY_LEVEL'
    if re.search(r'\b(direktor|director|general manager|chief|executive|baskan|vp|president|vice president)\b', clean):
        return 'MANAGER'
    if re.search(r'\b(manager|mudur|yonetici|lead|lider)\b', clean):
        return 'MANAGER'
    if re.search(r'\b(senior|kidemli|architect|mimar|usta)\b', clean):
        return 'SENIOR'
    return 'SPECIALIST'

def extract_job_city(raw):
    if not isinstance(raw, dict): return ''
    loc = raw.get('location')
    if isinstance(loc, dict):
        city = loc.get('city', '')
        if city:
            clean = re.sub(r'\(.*?\)', '', city).strip()
            return tr_normalize(clean).upper()
    return ''

TURKISH_CITIES = [
    'ISTANBUL', 'ANKARA', 'IZMIR', 'BURSA', 'ANTALYA', 'ADANA', 'KOCAELI', 'KONYA',
    'GAZIANTEP', 'KAYSERI', 'ESKISEHIR', 'MERSIN', 'SAMSUN', 'TEKIRDAG', 'DENIZLI',
    'SAKARYA', 'DIYARBAKIR', 'TRABZON', 'HATAY', 'ZONGULDAK', 'BALIKESIR', 'MALATYA',
    'SANLIURFA', 'ELAZIG', 'MANISA', 'MUGLA', 'KAHRAMANMARAS', 'YALOVA', 'ERZURUM',
    'AYDIN', 'SIVAS', 'KARABUK', 'KUTAHYA', 'TOKAT', 'VAN', 'CANAKKALE', 'BATMAN',
    'ERZINCAN', 'DUZCE', 'BOLU', 'RIZE', 'ORDU', 'GIRESUN', 'KARS', 'AGRI', 'IGDIR',
    'ARDAHAN', 'ARTVIN', 'AMASYA', 'CORUM', 'SINOP', 'KASTAMONU', 'BARTIN', 'CANKIRI',
    'KIRIKKALE', 'KIRSEHIR', 'NEVSEHIR', 'AKSARAY', 'NIGDE', 'KARAMAN', 'YOZGAT',
    'BOGAZICI', 'BODRUM', 'GEBZE', 'ISPARTA', 'BURDUR', 'AFYONKARAHISAR', 'USAK',
    'BILECIK', 'EDIRNE', 'KIRKLARELI', 'BINGOL', 'BITLIS', 'MUS', 'HAKKARI', 'SIRNAK',
    'SIIRT', 'MARDIN', 'ADIYAMAN', 'KILIS', 'OSMANIYE', 'BAYBURT', 'TUNCELI'
]

def get_cand_city(row):
    loc = tr_normalize(str(row.get('location', row.get('location_or_school', '')))).upper().strip()
    if loc == 'BODRUM':
        return 'MUGLA'
    if loc == 'GEBZE':
        return 'KOCAELI'
    for c in TURKISH_CITIES:
        if c == loc or c in loc:
            return c
    clean = tr_normalize(str(row.get('cv_text', ''))).upper()
    for c in TURKISH_CITIES:
        if len(c) > 3:
            if c in clean:
                return c
        else:
            if re.search(r'\b' + re.escape(c) + r'\b', clean):
                return c
    return 'UNSPECIFIED'

# -------------------------------------------------------------------------
# 1. TEMEL TOP-10 MODEL TESTİ (DCG, IDCG, NDCG VE SEKTÖR BAZLI SONUÇLAR)
# -------------------------------------------------------------------------
def calculate_ndcg(relevance_list, k=10):
    relevance_list = relevance_list[0:k]
    dcg = sum((2**rel - 1) / math.log2(i+2) for i, rel in enumerate(relevance_list))
    ideal_list = sorted(relevance_list, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i+2) for i, rel in enumerate(ideal_list))
    if idcg == 0.0:
        return 0.0, 0.0, 0.0
    return dcg, idcg, (dcg / idcg)

def is_industry_match(cand_ind, job_ind):
    return str(cand_ind).strip().upper() == str(job_ind).strip().upper()

# -------------------------------------------------------------------------
# 2. CEZA VE FİLTRELEME MEKANİZMALARI
# -------------------------------------------------------------------------
LEVEL_MAP = {
    'INTERN': 1,
    'STUDENT': 1,
    'TRAINEE': 1,
    'ENTRY_LEVEL': 2,
    'NEW_GRADUATE': 2,
    'ASSISTANT': 2,
    'ASSISTANT_SPECIALIST': 2,
    'JUNIOR': 2,
    'MID_LEVEL': 3,
    'SPECIALIST': 3,
    'SENIOR': 4,
    'SENIOR_SPECIALIST': 4,
    'TEAM_LEAD': 4,
    'MANAGER': 5,
    'EXECUTIVE': 5,
    'DIRECTOR': 5
}

def get_seniority_multiplier(cand_lvl_str, job_lvl_str):
    """
    Kıdem Seviyesi Çarpanı:
    - 1 kademe farka kadar (terfi / yükselme şansı veya doğal geçiş): Ceza uygulanmaz (1.00).
    - 2 veya daha fazla kademe fark: Ceza 2. kademeden itibaren başlar ((0.95)^(diff - 1)).
      * 1 fark (Örn: Senior -> Manager, Entry -> Specialist): 1.00 (Ceza yok)
      * 2 fark (Örn: Entry -> Senior, Specialist -> Manager): 0.95 (%5 ceza)
      * 3 fark (Örn: Entry -> Manager): 0.9025 (%10 ceza)
    """
    c_lvl = LEVEL_MAP.get(str(cand_lvl_str).upper(), 3)
    j_lvl = LEVEL_MAP.get(str(job_lvl_str).upper(), 0)
    if j_lvl == 0:
        return 1.0  # İlanda kıdem belirtilmemişse ceza uygulanmaz
    diff = abs(c_lvl - j_lvl)
    if diff <= 1:
        return 1.0  # 1 kademe farka kadar terfi / geçiş serbest
    return (0.95) ** (diff - 1)

def get_location_multiplier(cand_city, job_city, is_remote, cand_collar="WHITE"):
    """
    Lokasyon Çarpanı:
    - Remote ilanlar: 1.0 (Ceza yok, uzaktan çalışma)
    - Yerinde (OnSite / Hybrid) ilanlar:
      - Adayın şehri ile ilanın şehri uyuşuyorsa: 1.0 (Ceza yok)
      - Farklı şehir VEYA ilanda şehir belirtilmemişse:
        * Mavi yaka adayı için %15 ceza (0.85 çarpanı)
        * Beyaz yaka adayı için %5 ceza (0.95 çarpanı)
    """
    if is_remote:
        return 1.0
    c_c = tr_normalize(str(cand_city)).strip().upper()
    j_c = tr_normalize(str(job_city)).strip().upper()
    if c_c and j_c and c_c != 'UNSPECIFIED' and j_c not in ['', 'BELIRTILMEMIS', 'NONE'] and (c_c in j_c or j_c in c_c):
        return 1.0
    is_blue = str(cand_collar).strip().upper() in ["BLUE", "MAVI"]
    return 0.85 if is_blue else 0.95

def check_hard_filter(cand_col, job_col):
    if cand_col != job_col:
        return False
    return True

# -------------------------------------------------------------------------
# 3. BİLEŞİK EŞLEŞME KALİTESİ (CMQ) VE ÇOK BOYUTLU KADEMELİ NDCG
# -------------------------------------------------------------------------
def compute_all_qualities_matrix(df_candidates, df_jobs, similarity_matrix, cand_industries):
    """
    Bileşik Eşleşme Kalitesi (Composite Match Quality - CMQ) Matrisi:
    - Yaka (Hard Gate): Farklıysa doğrudan 0.0
    - %40 Rol / Mesleki Semantik Benzerlik (clip [0, 1])
    - %25 Sektör Eşitliği
    - %20 Kıdem Uyumu (1.0 - 0.20 * kademe farkı)
    - %15 Lokasyon Uyumu (Remote/Aynı Şehir: 1.0, Farklı Şehir: Beyaz 0.70 / Mavi 0.40)
    """
    cand_cols = np.array([str(c).strip().upper() for c in df_candidates['collar']])
    job_collars = np.array([str(c).strip().upper() for c in df_jobs['collar']])
    
    cand_inds = np.array([str(c).strip().upper() for c in cand_industries])
    job_inds = np.array([str(c).strip().upper() for c in df_jobs['industry']])

    cand_sens = np.array([LEVEL_MAP.get(str(s).strip().upper(), 3) for s in df_candidates['seniority']])
    job_sens = np.array([LEVEL_MAP.get(str(s).strip().upper(), 0) for s in df_jobs['position_level']])

    cand_cities = np.array([tr_normalize(str(c)).strip().upper() for c in df_candidates['city']])
    job_cities = np.array([tr_normalize(str(c)).strip().upper() for c in df_jobs['city']])
    job_remotes = np.array(df_jobs['is_remote'].tolist(), dtype=bool)

    # 1. Yaka (Collar): Hard Gate
    col_gate = (cand_cols[:, None] == job_collars[None, :]).astype(float)

    # 2. Rol: Raw similarity clipped to [0, 1]
    s_role = np.clip(similarity_matrix, 0.0, 1.0)

    # 3. Sektör: Industry match
    s_ind = (cand_inds[:, None] == job_inds[None, :]).astype(float)

    # 4. Kıdem: Level diff
    diff = np.abs(cand_sens[:, None] - job_sens[None, :])
    s_sen = np.where(job_sens[None, :] == 0, 1.0, np.maximum(0.0, 1.0 - 0.20 * diff))

    # 5. Lokasyon:
    same_city = (cand_cities[:, None] == job_cities[None, :]) & (cand_cities[:, None] != 'UNSPECIFIED') & (job_cities[None, :] != '')
    loc_full = job_remotes[None, :] | same_city
    is_blue = (cand_cols[:, None] == 'BLUE') | (cand_cols[:, None] == 'MAVI')
    pen = np.where(is_blue, 0.40, 0.70)
    s_loc = np.where(loc_full, 1.0, pen)

    # Ağırlıklı Bileşik Kalite Skoru:
    quality_matrix = col_gate * (0.40 * s_role + 0.25 * s_ind + 0.20 * s_sen + 0.15 * s_loc)
    return quality_matrix

def compute_graded_ndcg(top_k_indices, cand_all_q, k=10):
    """
    Çok Boyutlu Kademeli NDCG:
    - rel = 5.0 * Quality
    - DCG = sum((2^rel - 1) / log2(i + 2))
    - IDCG = all_qualities'in en yüksek ilk k değerinden hesaplanan ideal DCG
    """
    top_q = cand_all_q[top_k_indices[:k]]
    rel = 5.0 * top_q
    dcg = float(np.sum((2.0 ** rel - 1.0) / np.log2(np.arange(len(top_q)) + 2)))
    
    ideal_q = np.sort(cand_all_q)[::-1][:k]
    ideal_rel = 5.0 * ideal_q
    idcg = float(np.sum((2.0 ** ideal_rel - 1.0) / np.log2(np.arange(len(ideal_q)) + 2)))
    
    if idcg <= 0.0:
        return 0.0
    return float((dcg / idcg) * 100.0)

MODEL_PATH = r'c:\Users\akcic\Desktop\konteyner\cv_scoring\yeni_finetune\multilingual-e5-small'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r'c:\Users\akcic\Desktop\konteyner\cv_scoring\yeni_finetune\egitim_e5'
model_path = MODEL_PATH

def main():
    print("Eğitilmiş E5 Modeli yükleniyor...")
    model = SentenceTransformer(MODEL_PATH)
    model.max_seq_length = 512

    test_path = r'c:\Users\akcic\Desktop\konteyner\cv_scoring\yeni_finetune\veriseti\test.csv'
    df_test = pd.read_csv(test_path)
    df_candidates = df_test.drop_duplicates(subset=['cv_id']).copy().reset_index(drop=True)

    df_candidates['cv_clean_text'] = ("query: " + df_candidates['cv_text'].apply(clean_text))
    if 'NaturalTitle' in df_candidates.columns:
        cand_titles = df_candidates['NaturalTitle'].fillna('').astype(str).str.strip()
    else:
        cand_titles = df_candidates['cv_text'].apply(extract_cand_title)
    def normalize_collar(c):
        s = str(c).upper().strip()
        if 'BEYAZ' in s or 'WHITE' in s:
            return 'WHITE'
        if 'MAVI' in s or 'BLUE' in s:
            return 'BLUE'
        return 'WHITE'

    if 'collar' in df_candidates.columns:
        df_candidates['collar'] = df_candidates['collar'].apply(normalize_collar)
    else:
        df_candidates['collar'] = cand_titles.apply(classify_collar).apply(normalize_collar)
    df_candidates['seniority'] = cand_titles.apply(extract_cand_seniority)

    print(f"Teste sokulacak aday sayısı: {len(df_candidates)}")

    # Veritabanından İlanlar ve Üstverilerin Çekilmesi
    engine = alc.create_engine('postgresql://postgres:denemeamacli@localhost:5432/postgres')
    df_jobs = pd.read_sql("SELECT id, title, company_name, description, industry, position_level, mode_of_operation, raw_data FROM job_advertisements ORDER BY id", engine)
    df_jobs['collar'] = df_jobs['title'].apply(classify_collar).apply(normalize_collar)
    df_jobs['city'] = df_jobs['raw_data'].apply(extract_job_city)
    df_jobs['is_remote'] = df_jobs['mode_of_operation'].astype(str).str.lower() == 'remote'

    df_candidates['city'] = df_candidates.apply(get_cand_city, axis=1)

    df_jobs['job_clean_text'] = (
        "passage: İş İlanı: " + df_jobs['title'].astype(str) + 
        " Firma: " + df_jobs['company_name'].fillna('') +
        " Açıklama: " + df_jobs['description'].astype(str).str[:2000]
    ).apply(clean_text)

    print(f"Veri tabanından çekilen ilan sayısı: {len(df_jobs)}")

    print("\nVektörler çıkarılıyor...")
    job_embeddings = model.encode(df_jobs['job_clean_text'].tolist(), show_progress_bar=True, normalize_embeddings=True)
    cv_embeddings = model.encode(df_candidates['cv_clean_text'].tolist(), show_progress_bar=True, normalize_embeddings=True)

    similarity_matrix = np.dot(cv_embeddings, job_embeddings.T)

    # Bileşik Kalite Matrisi (CMQ) Hesaplanması (vektörize, çok boyutlu)
    if 'cv_industry' in df_candidates.columns:
        cand_industries = df_candidates['cv_industry'].tolist()
    elif 'Category' in df_candidates.columns:
        cand_industries = df_candidates['Category'].tolist()
    else:
        cand_industries = ['UNSPECIFIED'] * len(df_candidates)
    job_industries = df_jobs['industry'].tolist()
    toplam_aday_sayisi = len(cand_industries)

    quality_matrix = compute_all_qualities_matrix(df_candidates, df_jobs, similarity_matrix, cand_industries)

    all_dcg, all_idcg, all_ndcg, all_precisions = [], [], [], []
    all_top10_cmq, all_top10_graded_ndcg = [], []
    top1_correct = 0

    industry_ndcg = defaultdict(list)
    industry_precision = defaultdict(list)

    print("\nDeğerlendirme başlatılıyor...")

    for i in range(toplam_aday_sayisi):
        cand_ind = str(cand_industries[i]).strip().upper()
        top10_job_indices = np.argsort(similarity_matrix[i])[::-1][:10]
        
        # Ham model başarısı: Sektör eşitliği (2: tam eşleşme, 0: uyumsuz)
        relevance_list = [
            2 if is_industry_match(cand_ind, job_industries[j]) else 0
            for j in top10_job_indices
        ]
        dcg, idcg, ndcg = calculate_ndcg(relevance_list, k=10)
        
        all_dcg.append(dcg)
        all_idcg.append(idcg)
        all_ndcg.append(ndcg)
        industry_ndcg[cand_ind].append(ndcg)

        # Bileşik Kalite ve Çok Boyutlu Kademeli NDCG
        top10_q = quality_matrix[i, top10_job_indices]
        all_top10_cmq.append(np.mean(top10_q) * 100.0)
        all_top10_graded_ndcg.append(compute_graded_ndcg(top10_job_indices, quality_matrix[i], k=10))
        
        if relevance_list[0] > 0:
            top1_correct += 1
            
        dogru_ilan_sayisi = sum(1 for r in relevance_list if r > 0)
        prec = dogru_ilan_sayisi / 10.0
        all_precisions.append(prec)
        industry_precision[cand_ind].append(prec)

    avg_dcg = np.mean(all_dcg)
    avg_idcg = np.mean(all_idcg)
    avg_ndcg = np.mean(all_ndcg) * 100.0

    top1_acc = (top1_correct / toplam_aday_sayisi) * 100.0
    avg_precision = np.mean(all_precisions) * 100.0

    max_possible_dcg = sum((2**2 - 1) / math.log2(i + 2) for i in range(10))
    weighted_accuracy = (avg_dcg / max_possible_dcg) * 100.0

    print("\nI)CV YE UYGUN İLAN EŞLEŞTİRME TEST SONUÇLARI\nModel: multilingual-e5-small\nTOP-10 metrikleri:")
    print("Öneri Doğruluğu (Precision@10): %%%.2f" % avg_precision)
    print("Ağırlıklı Öneri Doğruluğu (Accuracy): %%%.2f" % weighted_accuracy)
    print("Top-1 Tam İsabet (Sektör Birebir): %%%.2f" % top1_acc)
    print("Maksimum Alınabilecek DCG (Teorik Tavan): %.2f" % max_possible_dcg)
    print("Ortalama IDCG (Havuz İdeal Tavanı): %.2f" % avg_idcg)
    print("Ortalama DCG (Modelin Aldığı Puan): %.2f" % avg_dcg)
    print("Ortalama NDCG@10: %%%.2f" % avg_ndcg)
    print("Bileşik Kalite İndeksi (CMQ@10): %%%.2f" % np.mean(all_top10_cmq))
    print("Çok Boyutlu Kademeli NDCG@10: %%%.2f" % np.mean(all_top10_graded_ndcg))

    sektor_sonuclari = []
    for sektor in industry_ndcg:
        puanlar = industry_ndcg[sektor]
        dogruluklar = industry_precision[sektor]
        sektor_sonuclari.append((np.mean(dogruluklar) * 100.0, np.mean(puanlar) * 100.0, sektor, len(puanlar)))

    sektor_sonuclari.sort(reverse=True)
    print("\nSektör Bazlı Sonuçlar (Doğruluk & NDCG):")
    for acc, ndcg, sektor, n in sektor_sonuclari:
        print("%s (%d aday): Doğruluk: %%%.2f | NDCG: %%%.2f" % (sektor, n, acc, ndcg))

    def run_evaluation(k=5, filter_mode="soft_rerank"):
        all_precisions = []
        all_cmq = []
        all_graded_ndcg = []
        top1_correct = 0

        cand_cols = df_candidates['collar'].tolist()
        job_collars = df_jobs['collar'].tolist()
        cand_seniorities = df_candidates['seniority'].tolist()
        job_seniorities = df_jobs['position_level'].tolist()
        cand_cities = df_candidates['city'].tolist()
        job_cities = df_jobs['city'].tolist()
        job_remotes = df_jobs['is_remote'].tolist()

        for i in range(toplam_aday_sayisi):
            cand_ind = str(cand_industries[i]).strip().upper()
            cand_col = cand_cols[i]
            cand_sen = cand_seniorities[i]
            cand_city = cand_cities[i]
            sims = similarity_matrix[i].copy()
            
            for j in range(len(df_jobs)):
                if filter_mode != "raw":
                    # 1. Yaka Uyuşmazlığı Cezası: -%25 (0.75 çarpanı)
                    collar_mult = 0.75 if cand_col != job_collars[j] else 1.0
                    
                    if filter_mode == "collar":
                        sims[j] = sims[j] * collar_mult
                    elif filter_mode == "soft_rerank":
                        # 2. Kıdem (0.95^n) ve Lokasyon (%5 Beyaz, %15 Mavi) Yüzdelik Cezaları
                        sen_mult = get_seniority_multiplier(cand_sen, job_seniorities[j])
                        loc_mult = get_location_multiplier(cand_city, job_cities[j], job_remotes[j], cand_col)
                        sims[j] = sims[j] * collar_mult * sen_mult * loc_mult
                        
            top_k_indices = np.argsort(sims)[::-1][:k]
            
            relevance_list = []
            for j in top_k_indices:
                j_ind = str(job_industries[j]).strip().upper()
                j_col = job_collars[j]
                match_ind = is_industry_match(cand_ind, j_ind)
                if filter_mode == "raw":
                    rel = 1 if match_ind else 0
                else:
                    rel = 1 if (match_ind and cand_col == j_col) else 0
                relevance_list.append(rel)
            
            if relevance_list[0] > 0:
                top1_correct += 1
                
            dogru_ilan_sayisi = sum(1 for r in relevance_list if r > 0)
            all_precisions.append(dogru_ilan_sayisi / float(k))

            # Bileşik Kalite ve Kademeli NDCG
            top_k_q = quality_matrix[i, top_k_indices]
            all_cmq.append(np.mean(top_k_q) * 100.0)
            all_graded_ndcg.append(compute_graded_ndcg(top_k_indices, quality_matrix[i], k=k))

        top1_acc = (top1_correct / toplam_aday_sayisi) * 100.0
        avg_precision = np.mean(all_precisions) * 100.0

        return {
            'precision': avg_precision,
            'top1': top1_acc,
            'cmq': np.mean(all_cmq),
            'graded_ndcg': np.mean(all_graded_ndcg)
        }

    print("\nA)FİLTRE DEĞERLENDİRME RAPORU")

    for k in [3, 5]:
        raw_res = run_evaluation(k=k, filter_mode="raw")
        collar_res = run_evaluation(k=k, filter_mode="collar")
        soft_res = run_evaluation(k=k, filter_mode="soft_rerank")
        
        print(f"\n--- TOP-{k} ÖNERİ LİSTESİ ---")
        print(f"Precision@{k} (Raw)           : %{raw_res['precision']:.2f}")
        print(f"Precision@{k} (Collar Filter) : %{collar_res['precision']:.2f}")
        print(f"Precision@{k} (All Filters)   : %{soft_res['precision']:.2f}")
        print(f"Top-1 Doğruluk (All filters)  : %{soft_res['top1']:.2f}")
        print(f"Bileşik Kalite İndeksi (CMQ@{k}): %{soft_res['cmq']:.2f}")
        print(f"Çok Boyutlu Kademeli NDCG@{k}: %{soft_res['graded_ndcg']:.2f}")

if __name__ == '__main__':
    main()
