"""
Готовит data_dir для AIDE-агента из сырых файлов KuaiRand-Pure.

Зачем этот скрипт, а не просто скормить агенту ./KuaiRand-Pure/data:
  - log_standard_4_22_to_5_08_pure.csv физически содержит и valid (20220422-20220428),
    и test (20220429-20220508) строки, включая long_view и все остальные outcome-колонки.
    Промпт-инструкция "не смотри в test" агента не останавливает - он умеет читать файлы.
    Этот скрипт вырезает test построчно и кладёт его ОТДЕЛЬНО от data_dir агента.
  - video_features_statistic_pure.csv вообще не копируется агенту: это агрегаты вида
    long_time_play_cnt/long_time_play_user_num и т.п., посчитанные по всему логу видео
    (то есть включая test-период). Это прямая утечка таргета через видео-агрегат,
    которую построчная маскировка test.csv не ловит.

Результат:
  <out>/train.csv                 20220408-20220421, все колонки (включая long_view
                                   и is_click/is_like/... - можно для multi-task)
  <out>/valid.csv                 20220422-20220428, все колонки
  <out>/test.csv                  20220429-20220508, БЕЗ outcome-колонок (см. OUTCOME_COLS)
  <out>/sample_submission.csv     формат отправки (см. submit.py в корне репо)
  <out>/log_random_valid.csv      20220422-20220428 срез случайного лога (для unbiased
                                   валидации, направление #7 из README); test-часть
                                   случайного лога никуда не пишется вообще
  <out>/user_features_pure.csv    статические user-фичи, копия без изменений
  <out>/video_features_basic_pure.csv  статические video-фичи (без статистик!), копия
  <out>/evaluate.py               НЕИЗМЕНЁННАЯ копия официального скорера
  <out>/data_dictionary.md        человекочитаемый словарь колонок + явные warning'и

  <held>/test_labels.csv          row_id,user_id,video_id,long_view для test-периода.
                                   НЕ класть в data_dir агента. Используется один раз,
                                   отдельно от агента, для финального скоринга сабмита
                                   (join по row_id/user_id/video_id).

Запуск (один раз, до старта AIDE):
    python3 prepare_data.py
"""
import csv, os, argparse, shutil

SPLITS = {'train': (20220408, 20220421), 'valid': (20220422, 20220428), 'test': (20220429, 20220508)}

# Колонки лога, которые являются исходом взаимодействия (в т.ч. сам таргет long_view).
# Всё это убирается из test.csv - агент не должен видеть исход того, что скорится.
# duration_ms сознательно ОСТАВЛЕН: baseline.py/data.py в родительском репо уже
# используют его как input-фичу (dur_bucket) на всех сплитах, включая test - таков
# зафиксированный протокол этого стартер-кита. Если у организаторов реальный hidden
# test устроен иначе (duration_ms там недоступен) - это надо сверить отдельно.
OUTCOME_COLS = {
    'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
    'play_time_ms', 'profile_stay_time', 'comment_stay_time', 'is_profile_enter',
    'long_view',
}

def in_range(date, lo_hi):
    lo, hi = lo_hi
    return lo <= date <= hi

def write_train_only(path, out_dir):
    with open(path, newline='') as fh:
        r = csv.DictReader(fh)
        fields = r.fieldnames
        with open(os.path.join(out_dir, 'train.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            n = 0
            for row in r:
                if in_range(int(row['date']), SPLITS['train']):
                    w.writerow(row); n += 1
    return n

def write_valid_test(path, out_dir, held_dir):
    with open(path, newline='') as fh:
        r = csv.DictReader(fh)
        fields = r.fieldnames
        test_cols = [c for c in fields if c not in OUTCOME_COLS]
        with open(os.path.join(out_dir, 'valid.csv'), 'w', newline='') as vf, \
             open(os.path.join(out_dir, 'test.csv'), 'w', newline='') as tf, \
             open(os.path.join(held_dir, 'test_labels.csv'), 'w', newline='') as hf:
            vw = csv.DictWriter(vf, fieldnames=fields); vw.writeheader()
            tw = csv.DictWriter(tf, fieldnames=['row_id'] + test_cols); tw.writeheader()
            hw = csv.DictWriter(hf, fieldnames=['row_id', 'user_id', 'video_id', 'long_view']); hw.writeheader()
            n_valid = n_test = 0
            for row in r:
                d = int(row['date'])
                if in_range(d, SPLITS['valid']):
                    vw.writerow(row); n_valid += 1
                elif in_range(d, SPLITS['test']):
                    tw.writerow({'row_id': n_test, **{c: row[c] for c in test_cols}})
                    hw.writerow({'row_id': n_test, 'user_id': row['user_id'],
                                 'video_id': row['video_id'], 'long_view': row['long_view']})
                    n_test += 1
    return n_valid, n_test

def write_random_valid_only(path, out_dir):
    """Направление #7 из README: случайный (unbiased) лог как доп. валидация.
    Пишем только valid-часть; test-часть случайного лога не пишем никуда."""
    with open(path, newline='') as fh:
        r = csv.DictReader(fh)
        fields = r.fieldnames
        with open(os.path.join(out_dir, 'log_random_valid.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            n = 0
            for row in r:
                if in_range(int(row['date']), SPLITS['valid']):
                    w.writerow(row); n += 1
    return n

def write_sample_submission(out_dir, n_test):
    with open(os.path.join(out_dir, 'test.csv'), newline='') as fh:
        first = next(csv.DictReader(fh))
    with open(os.path.join(out_dir, 'sample_submission.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row_id', 'user_id', 'video_id', 'score'])
        w.writerow([first['row_id'], first['user_id'], first['video_id'], 0.0])
        w.writerow(['...', '...', '...', '...'])
    print(f"  sample_submission.csv: пример строки, реально в test.csv {n_test:,d} строк")

DATA_DICT = """# Data dictionary / известные риски

## Файлы

| файл | период | что внутри |
|---|---|---|
| `train.csv` | 20220408-20220421 | полный лог, все колонки, включая `long_view` и остальные outcome-лейблы |
| `valid.csv` | 20220422-20220428 | то же самое, полный лог с лейблами - используйте для итеративной обратной связи |
| `test.csv` | 20220429-20220508 | **без** `long_view`/`is_click`/`is_like`/`is_follow`/`is_comment`/`is_forward`/`is_hate`/`play_time_ms`/`profile_stay_time`/`comment_stay_time`/`is_profile_enter` - это hidden test, лейблы физически не выданы |
| `sample_submission.csv` | - | формат ответа: `row_id,user_id,video_id,score` |
| `log_random_valid.csv` | 20220422-20220428 | случайный (unbiased) лог, valid-часть - опционально для доп. валидации |
| `user_features_pure.csv` | статика | user-side фичи, без привязки к периоду - безопасны |
| `video_features_basic_pure.csv` | статика | video-side фичи (author/type/music/...) - безопасны |
| `evaluate.py` | - | официальный скорер. **Не редактировать.** Всегда `from evaluate import evaluate` |

## ⚠️ Явно исключено

`video_features_statistic_pure.csv` **не включён** в этот data_dir вообще. Это агрегаты
вида `long_time_play_cnt`, `play_cnt`, `like_cnt` и т.п. по каждому видео, посчитанные по
всему логу видео (train+valid+test). Использование этого файла как фичи - прямая утечка
таргета через видео-агрегат, даже при том, что `test.csv` замаскирован построчно.
Если хотите похожий сигнал - считайте видео-агрегаты **сами**, только по `train.csv`.

## row_id

`row_id` в `test.csv` - 0-индексированный номер строки внутри test-периода (после
фильтрации по дате, в порядке исходного файла). `(user_id, video_id)` не уникальны как
ключ (дубликаты пар есть у части пользователей) - используйте `row_id` как primary key
в сабмите.
"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='../KuaiRand-Pure/data')
    ap.add_argument('--out', default='./data')
    ap.add_argument('--held', default='../held_out_test')
    ap.add_argument('--evaluate_py', default='../evaluate.py')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.held, exist_ok=True)

    n_train = write_train_only(os.path.join(a.src, 'log_standard_4_08_to_4_21_pure.csv'), a.out)
    print(f"train.csv: {n_train:,d} строк")

    n_valid, n_test = write_valid_test(os.path.join(a.src, 'log_standard_4_22_to_5_08_pure.csv'), a.out, a.held)
    print(f"valid.csv: {n_valid:,d} строк")
    print(f"test.csv:  {n_test:,d} строк (outcome-колонки вырезаны)")
    print(f"{a.held}/test_labels.csv: {n_test:,d} строк - НЕ отдавать агенту")

    n_rand = write_random_valid_only(os.path.join(a.src, 'log_random_4_22_to_5_08_pure.csv'), a.out)
    print(f"log_random_valid.csv: {n_rand:,d} строк")

    write_sample_submission(a.out, n_test)

    for fn in ('user_features_pure.csv', 'video_features_basic_pure.csv'):
        shutil.copy(os.path.join(a.src, fn), os.path.join(a.out, fn))
        print(f"скопирован {fn} (без изменений)")

    shutil.copy(a.evaluate_py, os.path.join(a.out, 'evaluate.py'))
    print("скопирован evaluate.py (без изменений)")

    with open(os.path.join(a.out, 'data_dictionary.md'), 'w') as f:
        f.write(DATA_DICT)
    print("написан data_dictionary.md")

    print("\nЯВНО НЕ скопировано: video_features_statistic_pure.csv (агрегаты, утечка таргета)")
