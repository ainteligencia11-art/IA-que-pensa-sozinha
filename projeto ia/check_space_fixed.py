# check_space_fixed.py
import os
import ctypes

def check_real_disk_space():
    print("💾 VERIFICAÇÃO REAL DE ESPAÇO EM DISCO...")
    
    # Método 1: Usando ctypes (mais preciso)
    free_bytes = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p("C:\\"), None, None, ctypes.pointer(free_bytes))
    free_gb = free_bytes.value / (1024**3)
    print(f"📊 Espaço livre em C: (ctypes): {free_gb:.2f} GB")
    
    # Método 2: Usando shutil (fallback)
    try:
        import shutil
        total, used, free = shutil.disk_usage("C:\\")
        print(f"📊 Espaço livre em C: (shutil): {free / (1024**3):.2f} GB")
    except:
        pass
    
    # Método 3: Usando os.statvfs (para Windows moderno)
    try:
        stat = os.statvfs("C:\\")
        free_gb_stat = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"📊 Espaço livre em C: (statvfs): {free_gb_stat:.2f} GB")
    except:
        pass
    
    return free_gb

# Verificar
free_space = check_real_disk_space()

if free_space > 50:
    print("🎉 ESPAÇO SUFICIENTE! Podemos continuar.")
elif free_space > 10:
    print("⚠️  ESPAÇO MODERADO. Cuidado com downloads grandes.")
else:
    print("🚨 POUCO ESPAÇO! Libere mais espaço antes de continuar.")