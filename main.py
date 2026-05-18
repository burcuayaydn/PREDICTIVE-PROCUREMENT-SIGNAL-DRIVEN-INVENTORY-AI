import os
import subprocess
import sys


def run_project():
    print("🚀 RetailRocket Akıllı Stok Yönetimi Yapay Zekası Başlatılıyor...\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    pipeline_scripts = [
        "scripts/01_data_loading.py",
        "scripts/02_eda.py",
        "scripts/03_preprocessing.py",
        "scripts/04_statistical_analysis.py",
        "scripts/05_modelling.py",
        "scripts/06_inventory_optimization.py"
    ]

    for script in pipeline_scripts:
        script_path = os.path.join(base_dir, script)

        if os.path.exists(script_path):
            print(
                f"--------------------------------------------------\n"
                f"⏳ Şu an çalışıyor: {script}...\n"
                f"--------------------------------------------------"
            )
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=base_dir,
                capture_output=False,
                text=True
            )

            if result.returncode != 0:
                print(f"❌ {script} çalıştırılırken bir hata oluştu! Süreç durduruldu.")
                return
        else:
            print(f"⚠️ Dosya bulunamadı: {script_path}")

    print("\n🎉 Tüm süreç başarıyla tamamlandı! Çıktılar 'outputs/' klasörüne kaydedildi.")


if __name__ == "__main__":
    run_project()