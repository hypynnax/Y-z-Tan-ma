import cv2
import tkinter as tk
import sqlite3
import tensorflow as tf
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageTk

bilgiResmi = None
ytad = None
ykad = None
model = DeepFace.Facenet.loadModel('facenet_keras.h5')
yuz_bulma_algoritmasi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class YuzTanima:
    def yuzTanima():
        global ytadLabel, ytad, ytsoyadLabel, ytsoyad, ytyasLabel, ytyas, ytcinsiyetLabel, ytcinsiyet, kameraCiktisiyt, kapatButonu, kamerayt, kisiler
        Uygulama.temizle()
        Uygulama.bilgileriAl()
        window.title(" Yüz Tanıma")
        ytadLabel = tk.Label(bilgiFrame, text='AD :', foreground='#00A2E8', background='#000000')
        ytadLabel.place(relx=0.1, rely=0.3)
        ytad = tk.Label(bilgiFrame, text='', foreground='#00FF00', background='#000000')
        ytad.place(relx=0.3, rely=0.3)
        ytsoyadLabel = tk.Label(bilgiFrame, text='SOYAD :', foreground='#00A2E8', background='#000000')
        ytsoyadLabel.place(relx=0.1, rely=0.4)
        ytsoyad = tk.Label(bilgiFrame, text='', foreground='#00FF00', background='#000000')
        ytsoyad.place(relx=0.3, rely=0.4)
        ytyasLabel = tk.Label(bilgiFrame, text='YAŞ :', foreground='#00A2E8', background='#000000')
        ytyasLabel.place(relx=0.1, rely=0.5)
        ytyas = tk.Label(bilgiFrame, text='', foreground='#00FF00', background='#000000')
        ytyas.place(relx=0.3, rely=0.5)
        ytcinsiyetLabel = tk.Label(bilgiFrame, text='CİNSİYET :', foreground='#00A2E8', background='#000000')
        ytcinsiyetLabel.place(relx=0.1, rely=0.6)
        ytcinsiyet = tk.Label(bilgiFrame, text='', foreground='#00FF00', background='#000000')
        ytcinsiyet.place(relx=0.3, rely=0.6)
        kapatButonu = tk.Button(bilgiFrame, text="KAPAT", foreground='#FFFFFF', background='#000000', activeforeground='#FFFFFF', activebackground='#0F0F0F', width=20, height=2, command=YuzTanima.kapat)
        kapatButonu.place(relx=0.266, rely=0.85)
        kameraCiktisiyt = tk.Canvas(window, width=841, height=535, highlightthickness=0)
        kameraCiktisiyt.place(relx=0.303, rely=0.083)
        kamerayt = cv2.VideoCapture(0)
        YuzTanima.kameraAc()

    def kapat():
        window.title(" Yüz Tanıma Uygulaması")
        Uygulama.temizle()

    def kameraAc():
        global model, kisiler, yuz_bulma_algoritmasi
        ret, goruntu = kamerayt.read()
        if ret:
            gri_foto = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
            yuzler = yuz_bulma_algoritmasi.detectMultiScale(gri_foto, 1.3, 5)
            for (x, y, w, h) in yuzler:
                yuz_resmi = goruntu[y:y+h, x:x+w]
                yuz_resmi = cv2.resize(yuz_resmi, (160, 160))
                yuz_resmi = cv2.cvtColor(yuz_resmi, cv2.COLOR_BGR2RGB)
                yuz_resmi = np.expand_dims(yuz_resmi, axis=0)
                yuz_tensor = tf.convert_to_tensor(yuz_resmi, dtype=tf.float32)
                embeddings = model(yuz_tensor)
                yuz_vektor = np.array(embeddings).reshape(-1)
                min_benzerlik = 999
                min_id = ""
                for id, bilgisi in kisiler.items():
                    benzerlik = np.linalg.norm(yuz_vektor - bilgisi['vektor'])
                    if benzerlik < min_benzerlik:
                        min_benzerlik = benzerlik
                        min_id = id
                print(min_benzerlik)
                if min_benzerlik < 0.5:
                    ytad.config(text=kisiler[min_id]['ad'])
                    ytsoyad.config(text=kisiler[min_id]['soyad'])
                    ytyas.config(text=kisiler[min_id]['yas'])
                    ytcinsiyet.config(text=kisiler[min_id]['cinsiyet'])                
                    cv2.line(goruntu, (x, y), (x+50, y), (255, 0, 0), 5)
                    cv2.line(goruntu, (x, y), (x, y+50), (255, 0, 0), 5)
                    cv2.line(goruntu, (x+w, y), (x+w, y+50), (255, 0, 0), 5)
                    cv2.line(goruntu, (x+w, y), (x+w-50, y), (255, 0, 0), 5)
                    cv2.line(goruntu, (x, y+h), (x+50, y+h), (255, 0, 0), 5)
                    cv2.line(goruntu, (x, y+h), (x, y+h-50), (255, 0, 0), 5)
                    cv2.line(goruntu, (x+w, y+h), (x+w, y+h-50), (255, 0, 0), 5)
                    cv2.line(goruntu, (x+w, y+h), (x+w-50, y+h), (255, 0, 0), 5)
                else:
                    ytad.config(text='')
                    ytsoyad.config(text='')
                    ytyas.config(text='')
                    ytcinsiyet.config(text='')
            image = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (841, 535))
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            kameraCiktisiyt.create_image(0, 0, anchor='nw', image=image)
            kameraCiktisiyt.photo = image
            window.after(15, YuzTanima.kameraAc)

class YuzKayit:
    def yuzKaydetmeBilgilendirme():
        global bilgiResmi, bilgiMetni, tamamButonu
        Uygulama.temizle()
        window.title(" Yüz Kayıt")
        image = cv2.imread("dogruPoz.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (240, 180))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        bilgiResmi = tk.Canvas(bilgiFrame, width=240, height=180, highlightbackground='#00A2E8', highlightthickness=0)
        bilgiResmi.place(relx=0.1, rely=0.05)
        bilgiResmi.create_image(0, 0, anchor='nw', image=image)
        bilgiResmi.photo = image
        text = "1.Yukardaki resim gibi üç resim çekin\n2.Çekilen resimlere ait bilgileri tam girin\n3.KAYDET butonuna basın\n4.Artık yüz tanıma işlemine geçilebilir.\nNot:Sağ ve sol profillerden resim koymak \nprogramın daha doğru çalışmasını sağlayacaktır."
        bilgiMetni = tk.Label(bilgiFrame, text=text, foreground='#FFFFFF', background='#000000')
        bilgiMetni.place(relx=0.08, rely=0.45)
        tamamButonu = tk.Button(bilgiFrame, text="TAMAM", foreground='#FFFFFF', background='#000000', activeforeground='#FFFFFF', activebackground='#0F0F0F', width=20, height=2, command=YuzKayit.yuzKayitEt)
        tamamButonu.place(relx=0.266, rely=0.8)

    def yuzKayitEt():
        global cekilenResimSayisi, cekilenResimSayisiLabel, ykadLabel, ykad, yksoyadLabel, yksoyad, ykyasLabel, ykyas, ykcinsiyetLabel, ykerkek, ykkiz, var, resimCekButonu, kayitButonu, kameraCiktisiyk, kamerayk, cekilenResimler
        Uygulama.temizle()
        cekilenResimSayisi = 0
        cekilenResimSayisiLabel = tk.Label(bilgiFrame, text='Çekilen Resim Sayısı :', foreground='#00A2E8', background='#000000')
        cekilenResimSayisiLabel.place(relx=0.1, rely=0.2)
        ykadLabel = tk.Label(bilgiFrame, text='AD :', foreground='#00A2E8', background='#000000')
        ykadLabel.place(relx=0.1, rely=0.3)
        ykad = tk.Entry(bilgiFrame, font='#000000')
        ykad.place(relx=0.32, rely=0.3)
        yksoyadLabel = tk.Label(bilgiFrame, text='SOYAD :', foreground='#00A2E8', background='#000000')
        yksoyadLabel.place(relx=0.1, rely=0.4)
        yksoyad = tk.Entry(bilgiFrame, font='#000000')
        yksoyad.place(relx=0.32, rely=0.4)
        ykyasLabel = tk.Label(bilgiFrame, text='YAŞ :', foreground='#00A2E8', background='#000000')
        ykyasLabel.place(relx=0.1, rely=0.5)
        ykyas = tk.Entry(bilgiFrame, font='#000000')
        ykyas.place(relx=0.32, rely=0.5)
        ykcinsiyetLabel = tk.Label(bilgiFrame, text='CİNSİYET :', foreground='#00A2E8', background='#000000')
        ykcinsiyetLabel.place(relx=0.1, rely=0.6)
        var = tk.IntVar()
        ykerkek = tk.Radiobutton(bilgiFrame, text="Erkek", variable=var, value=1, background='#000000', foreground='#FFFFFF', activebackground='#000000', activeforeground='#FFFFFF', selectcolor='#000000')
        ykerkek.place(relx=0.4, rely=0.6)
        ykkiz = tk.Radiobutton(bilgiFrame, text="Kız", variable=var, value=2, background='#000000', foreground='#FFFFFF', activebackground='#000000', activeforeground='#FFFFFF', selectcolor='#000000')
        ykkiz.place(relx=0.65, rely=0.6)
        resimCekButonu = tk.Button(bilgiFrame, text="Resim Çek", foreground='#FFFFFF', background='#000000', activeforeground='#FFFFFF', activebackground='#0F0F0F', width=20, height=2, command=YuzKayit.resimCek)
        resimCekButonu.place(relx=0.266, rely=0.75)
        kayitButonu = tk.Button(bilgiFrame, text="KAYDET", foreground='#FFFFFF', background='#000000', activeforeground='#FFFFFF', activebackground='#0F0F0F', width=20, height=2, command=YuzKayit.kayitEt)
        kayitButonu.place(relx=0.266, rely=0.85)
        kameraCiktisiyk = tk.Canvas(window, width=841, height=535, highlightthickness=0)
        kameraCiktisiyk.place(relx=0.303, rely=0.083)
        kamerayk = cv2.VideoCapture(0)
        YuzKayit.kameraAc()
        cekilenResimler=list()

    def kameraAc():
        global goruntu, ret
        ret, goruntu = kamerayk.read()
        if ret:
            image = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (841, 535))
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            kameraCiktisiyk.create_image(0, 0, anchor='nw', image=image)
            kameraCiktisiyk.photo = image
        window.after(15, YuzKayit.kameraAc)

    def resimCek():
        global cekilenResimSayisi, cekilenResimSayisiLabel, cekilenResimler, goruntu
        cekilenResimler.append(goruntu)
        cekilenResimSayisi += 1
        cekilenResimSayisiLabel.config(text="Çekilen Resim Sayısı : {}".format(cekilenResimSayisi))

    def kayitEt():
        global ret, model
        if (len(cekilenResimler) != 0) and (ykad.get() != "") and (yksoyad.get() != "") and (ykyas.get() != "") and (ykyas.get().isdigit()) and (var.get() != 0):        
            kamerayk.release()
            cv2.destroyAllWindows()
            kameraCiktisiyk.destroy()
            ret = False
            ad = ykad.get().capitalize()
            soyad = str.upper(yksoyad.get())
            yas = ykyas.get()
            cinsiyet = "Erkek" if var.get() == 1 else "Kız"
            veri = ad + "_" + soyad + "_" + yas + "_" + cinsiyet
            con = sqlite3.connect("resimlerinSayisi.db")
            cursor = con.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS kullanici(bilgileri, yuzVektor)")
            con.commit()
            for goruntu in cekilenResimler:
                gri_foto = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
                yuzler = yuz_bulma_algoritmasi.detectMultiScale(gri_foto, 1.3, 5)
                for (x, y, w, h) in yuzler:
                    yuz_resmi = goruntu[y:y+h, x:x+w]
                    yuz_resmi = cv2.resize(yuz_resmi, (160, 160))
                    yuz_resmi = cv2.cvtColor(yuz_resmi, cv2.COLOR_BGR2RGB)
                    yuz_resmi = np.expand_dims(yuz_resmi, axis=0)
                    yuz_tensor = tf.convert_to_tensor(yuz_resmi, dtype=tf.float32)
                    embeddings = model(yuz_tensor)
                    yuz_vektor = np.array(embeddings).reshape(-1)
                    cursor.execute("INSERT INTO kullanici VALUES (?, ?)", (veri, yuz_vektor))
            con.commit()
            con.close()
            window.title(" Yüz Tanıma Uygulaması")
            Uygulama.temizle()

class GorselArayuz:
    def __init__(self):
        global window, bilgiFrame
        window = tk.Tk(className=" Yüz Tanıma Uygulaması")
        window.configure(width=1286, height=645, background='#000000')
        window.geometry("+%d+%d" % (40, 30))
        window.iconbitmap(default='yuzTanima.ico')

        tk.Frame(window, width=150, height=50, background='#00A2E8').place(relx=0.031, rely=0.062)
        tk.Frame(window, width=140, height=40, background='#000000').place(relx=0.034, rely=0.069)
        tk.Button(window, text="Yüz Kaydı Yap", foreground='#FF0000', background='#000000', activeforeground='#FF0000', activebackground='#0F0F0F', width=19, height=2, command=YuzKayit.yuzKaydetmeBilgilendirme).place(relx=0.034, rely=0.069)

        tk.Frame(window, width=150, height=50, background='#00A2E8').place(relx=0.155, rely=0.062)
        tk.Frame(window, width=140, height=40, background='#000000').place(relx=0.159, rely=0.069)
        tk.Button(window, text="Yüz Tanımaya Başla", foreground='#00FF00', background='#000000', activeforeground='#00FF00', activebackground='#0F0F0F', width=19, height=2, command=YuzTanima.yuzTanima).place(relx=0.159, rely =0.069)

        tk.Frame(window, width=310, height=505, background='#00A2E8').place(relx=0.031, rely =0.155)
        bilgiFrame = tk.Frame(window, width=300, height=495, background='#000000')
        bilgiFrame.place(relx=0.034, rely =0.162)

        tk.Frame(window, width=871, height=565, background='#00A2E8').place(relx=0.291, rely=0.062)
        anaKontrolFrame = tk.Frame(window, width=851, height=545, background='#000000')
        anaKontrolFrame.place(relx=0.299, rely=0.077)
        tk.Frame(window, width=571, height=565, background='#000000').place(relx=0.409, rely=0.062)
        tk.Frame(window, width=871, height=265, background='#000000').place(relx=0.291, rely=0.295)

        window.mainloop()

class Uygulama:
    def __init__(self):
        arayuz = GorselArayuz()
    
    def temizle():
        cv2.destroyAllWindows()
        if ytad != None:
            ret=False
            ytadLabel.destroy()
            ytad.destroy()
            ytsoyadLabel.destroy()
            ytsoyad.destroy()
            ytyasLabel.destroy()
            ytyas.destroy()
            ytcinsiyetLabel.destroy()
            ytcinsiyet.destroy()
            kapatButonu.destroy()
            kameraCiktisiyt.destroy()
            kamerayt.release()
        if ykad != None:
            cekilenResimSayisiLabel.destroy()
            ykadLabel.destroy()
            ykad.destroy()
            yksoyadLabel.destroy()
            yksoyad.destroy()
            ykyasLabel.destroy()
            ykyas.destroy()
            ykcinsiyetLabel.destroy()
            ykkiz.destroy()
            ykerkek.destroy()
            resimCekButonu.destroy()
            kayitButonu.destroy()
            kameraCiktisiyk.destroy()
            kamerayk.release()
        if bilgiResmi != None:
            bilgiResmi.destroy()
            bilgiMetni.destroy()
            tamamButonu.destroy()
        
    def bilgileriAl():
        global kisiler
        kisiler={}      
        con = sqlite3.connect("resimlerinSayisi.db")
        cursor = con.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS kullanici(bilgileri, yuzVektor)")
        con.commit()
        cursor.execute("Select * from kullanici")
        bilgiler = cursor.fetchall()
        con.commit()
        con.close()
        for sira, bilgi in enumerate(bilgiler):
            veriler=bilgi[0].split('_')
            kisiVerileri={}
            kisiVerileri['ad']=veriler[0]
            kisiVerileri['soyad']=veriler[1]
            kisiVerileri['yas']=veriler[2]
            kisiVerileri['cinsiyet']=veriler[3]
            kisiVerileri['vektor']=np.frombuffer(bilgi[1], dtype=np.float32)
            kisiler[sira]=kisiVerileri
    
uygulama = Uygulama()