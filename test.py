"""from preprocess_tr import *

sentence = "BU BİR ÖRNEK CÜMLEDİR30."

sentence = sentence.replace('I', 'ı').lower()
#(sentence)

cleaned_sentence = clean_punctuation(sentence)
#("Cleaned sentence:", cleaned_sentence)

without_stopwords = remove_stop_words(cleaned_sentence)
#("Without stopwords:", without_stopwords)

alpha_turkish = keep_alpha_turkish(without_stopwords)
#("Alpha Turkish:", alpha_turkish)

stemmed_sentence = stemming(alpha_turkish)
#("Stemmed sentence:", stemmed_sentence)

a = preprocess_turkish(sentence)
#(a)"""

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from preprocess_tr import *
import operator
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Load Keras model
model = load_model('v2.3_general_class/model-neural-net.keras')
df2 = pd.read_csv('cleaned_data.csv')
tokenizer = Tokenizer(num_words=5000, lower=True) # lower : boolean. Whether to convert the texts to lowercase. , num_words : the maximum number of words to keep, based on word frequency.
tokenizer.fit_on_texts(df2['description'])

classes = ['bilgisayar',
 'cep telefonu',
 'bilgisayar bileşenleri',
 'küçük ev aletleri',
 'çevre birimleri',
 'tüketici elektroniği',
 'yazılım',
 'elektronik',
 'oyun - hobi',
 'aksesuar ürünleri',
 'tüketim malzemeleri',
 'aksesuar & sarf malz.',
 'bilgisayar tablet yazıcı']  # categories

# Utility function to get predictions using Neural Net model
def categoryPredictionNN(description):
    
    description = description.lower()
    description = clean_punctuation(description)
    description = keep_alpha_turkish(description)
    description = remove_stop_words(description)
    description = stemming(description)

    information = description

    sequences = tokenizer.texts_to_sequences([information])
    x = pad_sequences(sequences, maxlen=500)
    prediction = model.predict(x)

    predScores = [score for pred in prediction for score in pred]
    predDict = {}
    for cla, score in zip(classes, predScores):
        predDict[cla] = score

    sortedPredictions = sorted(predDict.items(), key=operator.itemgetter(1), reverse=True)[:10]
    print(sortedPredictions)
    return sortedPredictions


description = "Chipset Intel SoC Platform İşlemci Çekirdek Sayısı 2 İşlemci Sayısı 1 Turbo İşlemci Hızı 2.8 İşlemci Hızı 2.8 İşlemci Markası Intel® İşlemci Intel Celeron İşlemci Modeli Celeron® RAM Tipi DDR4 Maks. Desteklenen Bellek 4 RAM Bellek Boyutu 4 Ekran Boyutu (inç) 15.6 Çözünürlük 1366x768 Ekran Boyutu 39.624 cm / 15.6 inç Görüntü Oranı 04:03 Görüntü kalitesi Full-HD Ekran Tipi TN Ekran boyutu(cm) 39.624 Ekran Özellikleri 220nits Anti-glare Grafik Kartı UHD 600 Grafik Bellek Tipi shared-memory Paylaşımlı Grafik Bellek Paylaşımlı Gb Cinsinden Toplam Depolama Alanı 128 Sabit disk kapasitesi 128 Sabit disk tipi SSD Hard disk 1 SSD , 128 GB SSD Evet Sürücü Hayır Ağ Evet WİFİ Evet Bluetooth Evet Bağlantılar Wi-Fi, Bluetooth, Usb Ses Kartı High Definition (HD) Audio Mikrofon Evet İşletim Sistemi İçin Ek Bilgiler Windows 11 Home Ön Kamera Evet Batarya kapasitesi 42 Batarya Ömrü 11 Yükseklik (maks.) 17.9 Üretici Garantisi 2 Yıl Resmi Distribütör Garantilidir Ürün Tipi Laptop Renk (Üreticiye Göre) Gri Ağırlık 1.54 Derinlik 236 Yükseklik 17.9 Boyutlar (GxYxD) / Ağırlık 360.2 mm x 17.9 mm x 236 mm / - Ambalajlı Ağırlık 2.38 Genişlik 360.2 Ambalaj Genişliği 533 Ambalaj Yüksekliği 74 Ambalaj Derinliği 333 Çevre Ölçüsü 134.7 cm Kutu İçeriği Notebook, Güç Adaptörü Ambalaj Boyutu 533 mm / 74 mm / 333 mm Üretim Yeri Çin Grafik Kart Üreticisi:Intel Ekran kartı:UHD Graphics 600"
text1 ="iPhone 6 - iPhone 6 Plus Cep Telefonu Genel Özellikleri iPhone 6 yalnızca daha büyük değil, her açıdan daha gelişmiş. Daha büyük ve çok daha ince. Daha güçlü ve çok daha verimli. Kusursuz metal yüzeyi, yeni Retina HD ekranla mükemmel bir uyum içinde. Donanım ve yazılımın olağanüstü birlikteliği, kesintisiz bir form ve her anlamda daha üstün, yeni bir iPhone nesli yarattı. Şimdiye kadarki en büyük iPhone ve en ince iPhone 6'yı tasarlarken her malzemeyi ve unsuru en ince ayrıntısına kadar inceledik. Yumuşak hatlı ve kesintisiz bir tasarıma bu şekilde ulaştık. Şimdiye kadarki en ince ekranımız sayesinde form da inceldi. Düğmeler, kolay kullanmanız için en iyi şekilde yerleştirildi. Anodize alüminyum, paslanmaz çelik ve cam, kusursuz bir şekilde bir araya geldi. Sonuçta binlerce küçük ayrıntının birleşmesiyle, ortaya büyük bir şey çıktı. Hatta, iki büyük şey: iPhone 6 ve iPhone 6 Plus. Yalnızca daha büyük bir ekran değil,daha iyi bir ekran iPhone 6 ve iPhone 6 Plus'ı gördüğünüzde ilk dikkatinizi çeken şey, daha yüksek çözünürlüklü yeni Retina HD ekranların boyutu olabilir. Ama deneyiminizin bunun çok daha ötesinde olacağı kesin. Daha yüksek kontrast sağlayan yenilikler, daha geniş görüntüleme açılarında bile daha gerçekçi renkler sunan, çift etki alanlı pikseller ve geliştirilmiş polarizör. Ve karşınızda şimdiye kadarki en ince ve en gelişmiş Multi‑Touch ekranlar. Büyük güç ve yüksek verimlilik iPhone 6'da ikinci nesil 64 bit masaüstü sınıfı mimariye sahip A8 çip bulunur. M8 yardımcı hareket işlemcisi, aralarında yeni bir barometrenin de bulunduğu gelişmiş sensörleri sayesinde, hareketlerinizi etkili bir şekilde ölçerek bu olağanüstü gücü artırır. Daha iyi performans ve pil ömrü, iPhone 6’nızı daha uzun süre kullanmanızı ve çok daha fazla şey yapabilmenizi sağlar. Dünyanın fotoğrafa bakış açısı bu kamerayla değişti,şimdi sıra videoda iPhone ile her gün, diğer kameralardan daha çok fotoğraf çekiliyor. Çünkü, iSight kamera bu işi son derece kolay hale getiriyor. Ve dünyanın en popüler kamerası, 1.5 mikron piksel boyutunun ve ƒ/2.2 diyaframın yanı sıra herkesin harika fotoğraflar çekmesini sağlayacak yepyeni teknolojilerle donatıldı. Mükemmel videolar çekmeyi de kolay bir hale getirmek için; hızlandırılmış video modu, saniyede 60 kare 1080p HD video ve saniyede 240 kare ağır çekim video gibi yeni özellikler ekledik. Üstelik şimdi, çektiğiniz HD videoların muhteşem sonuçlarını geniş Retina HD ekranda izleyebilirsiniz. Daha hızlı kablosuz bağlantı daha fazla yerde Tüm dünya elinizin altındaymış gibi hissettiren süper hızlı bağlantı, iPhone 6’nın daha yüksek indirme hızlarına1 ulaşmasını ve tüm dünyadaki ağlara bağlanmasını sağlar. Ayrıca daha gelişmiş kablosuz teknolojileri destekleyerek performansı artırır ve sizin için önemli olan şeylerle ve kişilerle bağlantı kurabilmenize yardımcı olur. Güvenlik Parmağınızın ucunda Parmak iziniz, en güvenli parolanızdır. Çünkü her zaman yanınızdadır. Ve sadece size özeldir. Çığır açan Touch ID teknolojimiz, benzersiz bir parmak izi sensörü kullanarak telefonunuzun kilidini açmayı kolay ve güvenli hale getirdi. Üstelik iOS 8 ve Touch ID'deki yeni özellikler sayesinde, parmak izinizle şimdi çok daha fazlasına, daha hızlı erişebileceksiniz. Şimdiye kadarki en büyük iOS sürümü Büyük olan yalnızca iPhone 6 değil. Dünyanın en gelişmiş mobil işletim sistemi de çıtasını yükseltti. iOS 8, örneğin sağlık ve spor uygulamalarını kullanarak doktorunuzla iletişim kurmak gibi, daha önce yalnızca hayal edebildiğiniz şeyleri yapmanızı sağlayan özellik ve işlevlere sahip. Şimdi geliştiriciler, iOS 8'in olağanüstü yeni özelliklerini uygulamalarına taşıyor. Çünkü daha derinlemesine erişimleri ve daha fazla araçları var. Ve bunların hepsi, büyük Retina HD ekranda muhteşem görünüyor. Apple iPhone 6 Plus 16GB Uzay Gri Cep Telefonu (MGA82TU/A) Özellikleri RenkUzay Gri Dokunmatik EkranVar Ekran TipiIPS Led Ekran Ekran Çözünürlüğü1920 x 1080 Piksel Ekran Boyutu5.5 İşletim SistemiiOS 8 İşlemciA8 İşlemci Görüntülü GörüşmeVar 3G (Bağlantı Hızı)Var Dahili Hafıza16 Gb Arttırılabilir HafızaYok Wi-FiVar Çift HatYok Kamera8 Megapixel Kamera ZoomVar Kamera FlaşVar Video KayıtVar MesajVar Email desteğiVar GPS VAR GprsVar EdgeVar WapVar Bluetooth ÖzelliğiVar Radyo (FM)Yok Mp3Var TitreşimVar Bas KonuşVar AjandaVar AlarmVar Hesap MakinesiVar Boyutlar158,1 x 77,8 x 7,1 mm Ağırlık172 Gr Kutu İçeriğiCep Telefonu, Batarya, Şarj Aleti, Kablolu Kulaklık Garanti Süresi24 Ay APPLE TÜRKİYE tarafından garantilidir. Apple iPhone 6 Plus 16GB Uzay Gri Cep Telefonu (MGA82TU/A) Resimleri Ürüne resim ekleyin 0.10 TL değerinde incepuan kazanın! Apple iPhone 6 Plus 16GB Uzay Gri Cep Telefonu (MGA82TU/A) Fiyatları ve Ödeme Seçenekleri TaksitTaksit TutarıToplam Tutar 3 910,66 TL 2.732,00 TL 6 455,33 TL 2.732,00 TL 9 303,56 TL 2.732,00 TL TaksitTaksit TutarıToplam Tutar 2+4 455,33 TL 2.732,00 TL 3+4 390,28 TL 2.732,00 TL 9 309,63 TL 2.786,63 TL TaksitTaksit TutarıToplam Tutar 3 910,66 TL 2.732,00 TL 5 551,86 TL 2.759,32 TL 9 312,66 TL 2.813,95 TL TaksitTaksit TutarıToplam Tutar 3 910,66 TL 2.732,00 TL 6 459,89 TL 2.759,32 TL 9 306,59 TL 2.759,32 TL TaksitTaksit TutarıToplam Tutar 3 910,66 TL 2.732,00 TL 6 459,89 TL 2.759,32 TL 9 312,66 TL 2.813,95 TL TaksitTaksit TutarıToplam Tutar 3 910,66 TL 2.732,00 TL 6 455,33 TL 2.732,00 TL 9 303,56 TL 2.732,00 TL Kredi Kartı Tek Çekim %2 İNDİRİM2.677,36 TL Banka Havalesi / EFT ile %3 İNDİRİM2.650,04 TL Apple iPhone 6 Plus 16GB Uzay Gri Cep Telefonu (MGA82TU/A) ile en çok karşılaştırılan ürünler %23 Lg G3 D855 32 Gb Beyaz Cep Telefonu %23 Sony Xperia Z2 Beyaz Cep Telefonu %22 Samsung N9000Q Galaxy Note 3 Beyaz Cep Telefonu %19 Lg G3 D855 16 Gb Titan Cep Telefonu %13 Samsung Galaxy G900H S5 16 Gb Siyah Cep Telefonu Seçili Ürünleri Karşılaştır Neden incehesap.com ? Binlerce müşterimizin bilgisayar ve elektronik ürün ihtiyaçlarını bizden temin ediyor olması; küçük, orta ve büyük ölçekli işletmelerden bireysel müşterilere kadar geniş bir satış ağımızın olması, en uygun fiyatları siz değerli müşterilerimize sunmamız, alanında uzmanlaşmış kadro, hızlı sevkiyat, uygun ödeme seçenekleri ve güvenli alışveriş imkanı gibi birçok sebepten dolayı biz, tercih edebileceğiniz en iyi ve güvenilir online satış sitelerinden birisiyiz. Türkiye’nin önde gelen distribütörlerinin bizim için ne söylediklerini görmek için tıklayınız.LojistikMümkün olan en kısa sürede; yani birçok üründe siparişlerinizin aynı gününde teminini gerçekleştiriyor ve siparişlerinizi size gönderilmek üzere kargoya veriyoruz. Adetli ürün teminlerini de hızlı bir şekilde gerçekleştirebiliyoruz. Bunun yanı sıra hâlihazırda onbinlerce ürünü de kendi stoğumuzda bulunduruyoruz, bundan dolayı siparişlerinizi en hızlı ve güvenilir şekilde teslim eden firmalardan biri de biziz.Müşteri Hizmetleri Konusunda uzman bir kadroya sahip olan müşteri temsilcisi arkadaşlarımız sayesinde, ürünler ile ilgili bilgilere, talepleriniz ile ilgili cevaplara ve daha birçok konuda yardıma online veya telefon ile ulaşabilir, destek alabilirsiniz. Alacağınız ya da aldığınız ürünler ile ilgili her tür bilgiyi sunar, sadece sorularınızı cevaplamaz ayrıca bütçenize de en uygun ürünleri öneririz.Fiyat AvantajıDeneyimli satın alma ekibimiz size en uygun fiyatları verebilmek için her hafta binlerce farklı fiyatı kontrol ediyor. Eğer aynı ürünlerde bizim fiyatlarımızdan daha ucuzunu bulur, ürünü satın almadan önce bize bildirirseniz ilgili ürünün fiyatını da satabileceğimiz en düşük fiyata indiririz. ( Bu fiyatları bize bildirmek için [email protected] 'a mail atabilir, yada Müşteri Hizmetleri sayfamızdan iletişime geçebilirsiniz. )Ürün DanışmanıSize, alışverişinizi kolaylaştırmanız için çeşitli ürünlerde hizmete sunulmuş online danışmandır. Biz bu danışmanı sadece satış yapmak için değil memnun müşteri yaratmak için de oluşturduk. Ürünler ile ilgili çok az bilgiye sahip olsanız bile, Ürün Danışman’ımız sayesinde kolaylıkla ihtiyacınıza uygun ürünü bulabilirsiniz. Ürün Danışman’ını kullanmak tamamen ücretsiz olup, üstelik ürünü bizden alma zorunluluğu bulunmamaktadır. Ürün Danışman’ına ulaşmak için lütfen tıklayınız.Hem distribütör hem de tüketici gözüyle bakarızSadece çok iyi bir satış sitesi değil aynı zamanda en çok satılan birkaç ürünün Türkiye distribütörü ve birçok markanın da dağıtıcısıyız. Müşteri memnuniyetinin ürün kalitesinden çok daha önemli olduğunu biliyor ve bütün gücümüzü taleplerinizi karşılamak ve beklentilerinizi aşmak yönünde harcıyoruz.Keyifli alışverişler..."
evAleti = "Arzum ütü, 1. sınıf kaliteli malzemelerden imal edilerek yüksek verimlilik ve dayanıklılık sunar. Dikey buharlı ütü, damlama önleme özelliği sayesinde memnuniyet sağlar. Seramik tabanlı ütü, geleneksel modellerin yanı sıra kumaşları sararma ve lekelenmelere karşı korur. Arzum AR688 Claro seramik tabanlı ütü, uzun süreli kullanımda ağırlık hissini ortadan kaldırarak ergonomi sunar. 2400 W buharlı ütü, yüksek ısı teknolojisi sayesinde farklı kumaş türleri üzerinde etkilidir. Dikey şekilde kullanılabilen ütü modeli, asılı duran eşyalar için işlevsellik sağlar. Şok buhar kazanlı ütü, kırışıkları üst seviye ısı performansı sayesinde kısa sürede giderir. Kireç önleme teknolojisi ile geliştirilen Arzum AR688 Claro, ihtiyacınızı uzun yıllar karşılar. Sürekli buhar veren ütü sayesinde uzun süreli işlem gerektiren kumaşlarda bile kısa sürede ideal sonuçlar elde edebilirsiniz."
iphone ="Son teknolojiyle üretilen iPhone 13, ultra geniş 12 MP iki ana kamera ve 12 MP ön kamerayla profesyonel fotoğraflama ve video imkanı sunarken, sinematik mod kamera özelliğiyle derinlik efekti ve odak geçişi yapar. Geniş hacimli 128 GB kapasitesiyle fotoğraf, video ve uygulamalarınızı rahatlıkla depolar. Her anın vazgeçilmez aksesuarı siyah telefonlar, tarzınıza uyacak şekilde tasarlanmıştır. Dayanıklı tasarıma sahip Apple iPhone 13, havacılık ve uzay endüstrisi standartlarında bir malzeme olan alüminyum kenarlardan oluşur. Geniş kapsama alanı sunan 5G telefonlar, hızlı ve kesintisiz iletişim hizmeti sağlar. Kablosuz şarj olan telefonlar, aynı zamanda uzun pil kullanımı sağlayan yüksek teknolojiden oluşur. Modern çizgileriyle Apple iPhone 13, 128 GB suya ve toza dayanıklı olacak şekilde üretilmiştir. İşlevsel ve kompakt bir sistemden oluşan iOS telefonlar, farklı ihtiyaçlara uygun uygulama çeşitliliği yaratır. 4 GB RAM'e sahip telefon, 60 Hz ekran yenileme hızıyla kolay kullanım sağlar. 6.1 inç Super Retina XDR geniş ve ultra çözünürlüklü ekran ile üst düzey deneyim sunar. Üst düzey teknolojinin ürünü olan OLED ekran, parlak, net ve canlı renkleriyle kullanışlıdır. Akıllı performans özelliğine sahip NFC cep telefonu, kişisel güvenlik alanı oluşturur."
prediction = categoryPredictionNN(text1)