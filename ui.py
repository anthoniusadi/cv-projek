
from genericpath import exists
from tkinter import *
from tkinter import messagebox
import os
import datetime as dt
import pytz

master = Tk()
master.title('Handheld')
master.geometry('500x200')

def create_folder(name):
    try :
        folder = name
        exist = os.path.exists(folder)
        if not exist:
            os.makedirs(folder)
            print('diretory created')
    except FileExistsError:
        print('directory already exist')
    return folder
def save():
    indonesia_time = pytz.timezone('Asia/Jakarta')
    x = dt.datetime.now(indonesia_time)

    date_time = (x.strftime("%d%b%y_%H:%M:%S"))
    print(date_time)
    path_folder = create_folder(rekam_medis_entry.get())
    format_file = []
    # global nama_dokter_entry
    print(nama_dokter_entry.get())
    format_file.append('Nama Pasien : '+nama_pasien_entry.get())
    format_file.append('Umur : '+umur_entry.get())
    format_file.append('No. Rekam Medis : '+rekam_medis_entry.get())
    format_file.append('Nama Dokter : '+nama_dokter_entry.get())
    format_file.append('Nama RS : '+nama_rs_entry.get()) 
    with open(path_folder+'/'+'metadata_'+str(date_time)+'.txt','w') as f:
        f.write('\n'.join(format_file))
    messagebox.showinfo( "Save file", nama_pasien_entry.get() + ' Saved')
   

nama_pasien = Label(master, text='Nama Pasien').grid(row=0)
umur = Label(master, text='Umur').grid(row=1)
rekam_medis = Label(master, text='No.Rekam Medis').grid(row=2)
nama_dokter = Label(master, text='Nama Dokter').grid(row=3)
nama_rs = Label(master, text='Nama Rumah Sakit').grid(row=4)

b1 = Button(master, text ="Scan", command = save)

nama_pasien_entry = Entry(master)
umur_entry = Entry(master)
rekam_medis_entry = Entry(master)
nama_dokter_entry = Entry(master)
nama_rs_entry = Entry(master)

nama_pasien_entry.grid(row=0, column=1)
umur_entry.grid(row=1, column=1)
rekam_medis_entry.grid(row=2, column=1)
nama_dokter_entry.grid(row=3, column=1)
nama_rs_entry.grid(row=4, column=1)
b1.grid(row=5,column=1)

mainloop()