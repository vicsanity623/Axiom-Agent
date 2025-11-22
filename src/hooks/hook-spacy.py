from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("en_core_web_sm")
hiddenimports = collect_submodules("en_core_web_sm")
