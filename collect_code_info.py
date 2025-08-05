import os

# Пути к файлам с плюсами
files_with_plus = [
    "docker-compose.yml",
    "Dockerfile",
    "src/models/megnet.py",
    "src/server/api.py"
]

# Путь к выходному файлу
output_txt_path = "code_summary.txt"

# Открываем файл для записи
with open(output_txt_path, "w", encoding="utf-8") as out_file:
    for rel_path in files_with_plus:
        abs_path = os.path.abspath(rel_path)
        out_file.write(f"Имя файла: {os.path.basename(rel_path)}\n")
        out_file.write(f"Путь к файлу: {abs_path}\n")
        out_file.write("Содержимое файла:\n")

        try:
            with open(rel_path, "r", encoding="utf-8") as f:
                contents = f.read()
                out_file.write(contents)
        except Exception as e:
            out_file.write(f"[Ошибка чтения файла: {e}]")

        out_file.write("\n" + "-"*80 + "\n")

print(f"✅ Файл создан: {output_txt_path}")
