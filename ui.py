
from genericpath import exists
from multiprocessing.resource_sharer import stop
from pydoc import TextRepr
from tkinter import *
from tkinter import messagebox
import os
import datetime as dt
import pytz
# import live_cam_nolidar as live_cam
import live_cam
# 
master = Tk()
master.title('Handheld')
master.geometry('500x200')

def run(folder_path,format_name):
    live_cam.main(folder_path,format_name)

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
    # messagebox.showinfo( "Save file", nama_pasien_entry.get() + ' Saved\nTekan OK untuk melanjutkan proses SCAN')
    format_name = f'{rekam_medis_entry.get()}_{date_time}'
    run(rekam_medis_entry.get(),format_name)
def clear():
    global nama_pasien,umur,rekam_medis,nama_dokter,nama_rs
    nama_pasien_entry.delete(0,END)
    umur_entry.delete(0,END)
    rekam_medis_entry.delete(0,END)
    nama_dokter_entry.delete(0,END)
    nama_rs_entry.delete(0,END)

def stop():
    live_cam.stop()
    master.destroy()
   

nama_pasien = Label(master, text='Nama Pasien').grid(row=0)
umur = Label(master, text='Umur').grid(row=1)
rekam_medis = Label(master, text='No.Rekam Medis').grid(row=2)
nama_dokter = Label(master, text='Nama Dokter').grid(row=3)
nama_rs = Label(master, text='Nama Rumah Sakit').grid(row=4)

b1 = Button(master, text ="Scan", command = save)
b2 = Button(master, text = "Clear",command = clear)
b3 = Button(master, text = "Quit",command = stop)

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
b1.grid(row=5,column=0)
b2.grid(row=5 , column=1)
b3.grid(row=5,column=2)

mainloop()