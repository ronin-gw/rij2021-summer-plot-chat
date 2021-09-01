#!/usr/bin/env python3
import sys
import json
import re
import os.path
import pickle
from datetime import datetime, timezone, timedelta
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from operator import itemgetter
from copy import copy

from sudachipy import tokenizer, dictionary
import jaconv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.font_manager

TIMELINE = os.path.join(os.path.dirname(__file__), "timeline.pickle")
TIMEZONE = timezone(timedelta(hours=9), "JST")

matplotlib.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro", "Yu Gothic", "Meirio", "Takao", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]

# (ward to plot, line style, color)
RTA_EMOTES = (
    ("rtaClap", "-", "#ec7087"),
    ("rtaPray", "-", "#f7f97a"),
    ("rtaGl", "-", "#5cc200"),
    ("rtaGg", "-", "#ff381c"),
    ("rtaHatena", "-", "#ffb5a1"),
    ("rtaR", "-", "white"),
    (("rtaRedbull", "rtaRedbull2"), "-", "#98b0df"),
    # ("rtaRedbull2", "-", "#98b0df"),

    # ("rtaPog", "-.", "#f8c900"),
    (("rtaCry", "BibleThump"), "-.", "#5ec6ff"),
    # ("rtaHello", "-.", "#ff3291"),
    # ("rtaHmm", "-.", "#fcc7b9"),
    # ("rtaOko", "-.", "#d20025"),
    # ("rtaWut", "-.", "#d97f8d"),
    # ("rtaChan", "-.", "green"),
    # ("rtaKappa", "-.", "#ffeae2"),

    ("rtaPolice", "-.", "#7891b8"),
    ("rtaKabe", "-.", "#bf927a"),
    ("rtaListen", "-.", "#5eb0ff"),
    ("rtaSleep", "-.", "#ff8000"),
    # ("rtaCafe", "--", "#a44242"),
    # ("rtaDot", "--", "#ff3291"),

    # ("rtaIizo", ":", "#0f9619"),
    ("rtaBanana", ":", "#f3f905"),
    # ("rtaShogi", ":", "#c68d46"),
    ("rtaFrameperfect", ":", "#ff7401"),
    ("rtaPixelperfect", ":", "#ffa300"),
    # ("rtaShi", ":", "#8aa0ec"),
    # ("rtaGift", ":", "white"),
    ("rtaAnkimo", ":", "#f92218"),
    ("rtaMaru", ":", "#c80730"),
    ("rtaBatsu", ":", "#5aafdd"),
    ("rtaCheer", ":", "#ffbe00"),
    # ("rtaGogo", ":", "#df4f69"),
    # ("rtaPokan", ":", "#838187"),

    (("草", "ｗｗｗ", "LUL"), "--", "green"),
    ("無敵時間", "--", "red"),
    ("かわいい", "--", "#ff3291"),
    ("〜ケンカ", "--", "orange"),
    ("石油王", "--", "yellow"),
    (("Kappu", "カップ", "カレーメシ", "日清食品"), "--", "#f0cb39"),
    (("PokSuicune", "スイクン"), "--", "#c38cdc"),
    ("Squid2", "--", "#80d2b4"),
    ("ファイナル", "--", "gray"),
    ("サクラチル", "--", "#ffe0e0"),
)
# VOCABULARY = set(w for w, _, _, in RTA_EMOTES if isinstance(w, str))
# VOCABULARY |= set(chain(*(w for w, _, _, in RTA_EMOTES if isinstance(w, tuple))))

# (title, movie start time as timestamp, offset hour, min, sec)
GAMES = (
    ("始まりのあいさつ", 1628617360.7, 0, 12, 51),
    ("ポケットモンスター ソード&シールド", 1628617360.7, 0, 47, 13),
    ("ポケパークWii ～ピカチュウの大冒険～", 1628617360.7, 5, 37, 24),
    ("Inmost", 1628617360.7, 8, 4, 48),
    ("CUPHEAD", 1628617360.7, 9, 11, 45),
    ("ケモノヒーローズ", 1628617360.7, 10, 7, 21),
    ("LOVE 2: kuso", 1628617360.7, 10, 51, 59),
    ("Super Cable Boy", 1628617360.7, 11, 11, 32),
    ("Donut County", 1628617360.7, 12, 3, 10),
    ("10秒走 RETURNS", 1628617360.7, 12, 55, 42, "right"),
    ("ドラゴンクエストVIII 空と海と大地と呪われし姫君", 1628617360.7, 13, 24, 56),
    ("ソフィーのアトリエ\n～不思議な本の錬金術士～ DX", 1628617360.7, 20, 58, 32),
    ("Assault Spy", 1628617360.7, 24, 2, 10),
    ("ゼルダ無双 厄災の黙示録", 1628617360.7, 25, 1, 10),
    ("デスクリムゾン", 1628617360.7, 29, 12, 57),
    ("エアホッケー＠GAMEPACK", 1628617360.7, 30, 13, 33, "right"),
    ("ペプシマン", 1628617360.7, 30, 27, 53),
    ("ビューティフルジョー", 1628617360.7, 31, 10, 49),
    ("Phoenotopia: Awakening", 1628617360.7, 32, 0, 30),
    ("メタルスラッグ5", 1628617360.7, 34, 18, 1, "right"),
    ("テイルズオブデスティニー ディレクターズカット", 1628617360.7, 35, 5, 26),
    ("クラッシュ・バンディクー4 とんでもマルチバース", 1628761124.7, 0, 3, 50),
    ("ゼルダの伝説 神々のトライフォース", 1628761124.7, 2, 11, 47),
    ("星のカービィ スターアライズ", 1628761124.7, 3, 30, 4),
    ("星のカービィ ロボボプラネット", 1628761124.7, 5, 15, 44),
    ("メジャー Wii パーフェクトクローザー", 1628761124.7, 7, 18, 50, "right"),
    ("マリオテニスアドバンス", 1628761124.7, 7, 38, 52),
    ("野生動物運動会", 1628761124.7, 9, 27, 39),
    ("カニノケンカ -Fight Crab-", 1628761124.7, 10, 34, 11),
    ("Age of Empires III:\nDefinitive Edition", 1628761124.7, 11, 35, 42),
    ("デススマイルズメガブラックレーベル", 1628761124.7, 12, 27, 25),
    ("聖剣伝説3 TRIALS of MANA", 1628761124.7, 13, 37, 48),
    ("バレットウィッチ", 1628761124.7, 16, 52, 8),
    ("ASTRO's PLAYROOM", 1628761124.7, 17, 21, 52),
    ("SEKIRO: SHADOWS DIE TWICE", 1628761124.7, 17, 56, 59),
    ("Indivisible", 1628761124.7, 18, 44, 35),
    ("零～zero～", 1628761124.7, 21, 12, 16),
    ("Bloodstained: Curse of the Moon 2", 1628761124.7, 22, 46, 10),
    ("ロックマン4", 1628761124.7, 23, 42, 59),
    ("白き鋼鉄のX THE OUT OF GUNVOLT", 1628761124.7, 24, 52, 5, "right"),
    ("デモンズブレイゾン 魔界村 紋章編", 1628761124.7, 25, 31, 58),
    ("Golf It!", 1628761124.7, 26, 27, 52),
    ("Wild Guns", 1628761124.7, 27, 9, 13),
    ("ミスタードリラーアンコール", 1628761124.7, 27, 37, 12),
    ("はたらくUFO", 1628761124.7, 28, 17, 40),
    ("Dr.エッグマンの\nミーンビーン\nマシーン", 1628761124.7, 28, 54, 23),
    ("遊戯王ファイブディーズ ウィーリーブレイカーズ", 1628761124.7, 29, 16, 7),
    ("マリオカート64", 1628761124.7, 30, 10, 35),
    ("ときめきメモリアル", 1628761124.7, 30, 52, 42),
    ("首都高バトル01", 1628761124.7, 31, 54, 13),
    ("ギャラクティックストーム", 1628761124.7, 33, 45, 28),
    ("東方紺珠伝 ～ Legacy of Lunatic Kingdom.", 1628761124.7, 34, 22, 27),
    ("東方紅輝心", 1628761124.7, 35, 15, 56),
    ("Hollow Knight", 1628761124.7, 35, 45, 40),
    ("サイバーシャドウ", 1628761124.7, 36, 39, 33),
    ("スーパーマリオギャラクシー2", 1628761124.7, 37, 47, 4, "right"),
    ("ソニックアドベンチャー2", 1628761124.7, 41, 2, 0),
    ("バイオハザード\n（1996）", 1628761124.7, 41, 51, 43),
    ("バイオハザード4", 1628761124.7, 42, 55, 26),
    ("Hades", 1628761124.7, 44, 41, 47),
    ("ファイアーエムブレム\n蒼炎の軌跡", 1628925747.7, 0, 11, 16),
    ("Deathstate", 1628925747.7, 2, 24, 36),
    ("魔法使いハナビィ", 1628925747.7, 2, 40, 47),
    ("サルゲッチュ3", 1628925747.7, 3, 13, 20),
    ("バーニングレンジャー", 1628925747.7, 4, 30, 18),
    ("快速天使", 1628925747.7, 5, 10, 36),
    ("高橋名人の冒険島3", 1628925747.7, 5, 55, 55),
    ("ザ・グレイトバトルII ラストファイターツイン", 1628925747.7, 6, 23, 49),
    ("謎の村雨城", 1628925747.7, 7, 4, 21),
    ("俺の料理", 1628925747.7, 7, 25, 23),
    ("ワギャンパラダイス", 1628925747.7, 8, 5, 36),
    ("ドンキーコング(GB)", 1628925747.7, 8, 58, 37),
    ("XIゴ", 1628925747.7, 10, 51, 53),
    ("バトルガレッガ", 1628925747.7, 12, 3, 32),
    ("DanceDance\nRevolution\nSuperNOVA", 1628925747.7, 13, 24, 5),
    ("新・北海道4000km", 1628925747.7, 14, 0, 51),
    ("桃太郎電鉄～昭和 平成 令和も定番！～", 1628925747.7, 14, 51, 23),
    ("リングフィットアドベンチャー", 1628925747.7, 17, 22, 54, "right"),
    ("終わりのあいさつ", 1628925747.7, 17, 56, 50, "right")
)


class Game:
    def __init__(self, name, t, h, m, s, align="left"):
        self.name = name
        self.startat = datetime.fromtimestamp(t + h * 3600 + m * 60 + s)
        self.align = align


GAMES = tuple(Game(*args) for args in GAMES)

WINDOWSIZE = 1
WINDOW = timedelta(seconds=WINDOWSIZE)
AVR_WINDOW = 60
PER_SECONDS = 60
DPI = 200
ROW = 6
PAGES = 4
YMAX = 750
WIDTH = 3840
HEIGHT = 2160

FONT_COLOR = "white"
FRAME_COLOR = "white"
BACKGROUND_COLOR = "#3f6392"
FACE_COLOR = "#274064"
ARROW_COLOR = "#ffff79"


class Message:
    _tokenizer = dictionary.Dictionary().create()
    _mode = tokenizer.Tokenizer.SplitMode.C

    pns = ("無敵時間", "石油王", "かわいい")
    pn_patterns = (
        (re.compile("(.！){4}"), "○！○！○！○！"),
        (re.compile("[\u30A1-\u30FF]+ケンカ"), "〜ケンカ")
    )

    @classmethod
    def _tokenize(cls, text):
        return cls._tokenizer.tokenize(text, cls._mode)

    def __init__(self, raw):
        self.name = raw["author"]["name"]
        self.emotes = set() if "emotes" not in raw else set(e["name"] for e in raw["emotes"])
        self.datetime = datetime.fromtimestamp(int(raw["timestamp"]) // 1000000).replace(tzinfo=TIMEZONE)

        self.message = raw["message"]
        self.msg = set()

        message = self.message
        for emote in self.emotes:
            message = message.replace(emote, "")

        #
        match = re.match("(.！){4}", message)
        for pattern, replace in self.pn_patterns:
            match = pattern.match(message)
            if match:
                self.msg.add(replace)
                message.replace(match.group(), "")

        #
        for pn in self.pns:
            if pn in message:
                self.msg.add(pn)
                message = message.replace(pn, "")

        #
        message = jaconv.h2z(message)

        # (名詞 or 動詞) (+助動詞)を取り出す
        parts = []
        currentpart = None
        for m in self._tokenize(message):
            part = m.part_of_speech()[0]

            if currentpart:
                if part == "助動詞":
                    parts.append(m.surface())
                else:
                    self.msg.add(''.join(parts))
                    parts = []
                    if part in ("名詞", "動詞"):
                        currentpart = part
                        parts.append(m.surface())
                    else:
                        currentpart = None
            else:
                if part in ("名詞", "動詞"):
                    currentpart = part
                    parts.append(m.surface())

        if parts:
            self.msg.add(''.join(parts))

        #
        kusa = False
        for word in copy(self.msg):
            if set(word) & set(('w', 'ｗ')):
                kusa = True
                self.msg.remove(word)
        if kusa:
            self.msg.add("ｗｗｗ")

        message = message.strip()
        if not self.msg and message:
            self.msg.add(message)

    def __len__(self):
        return len(self.msg)

    @property
    def words(self):
        return self.msg | self.emotes


def _parse_chat(paths):
    messages = []
    for p in paths:
        with open(p) as f, Pool() as pool:
            j = json.load(f)
            messages += list(pool.map(Message, j, len(j) // pool._processes))

    timeline = []
    currentwindow = messages[0].datetime.replace(microsecond=0) + WINDOW
    _messages = []
    for m in messages:
        if m.datetime <= currentwindow:
            _messages.append(m)
        else:
            timeline.append((currentwindow, *_make_timepoint(_messages)))
            while True:
                currentwindow += WINDOW
                if m.datetime <= currentwindow:
                    _messages = [m]
                    break
                else:
                    timeline.append((currentwindow, 0, Counter()))

    if _messages:
        timeline.append((currentwindow, *_make_timepoint(_messages)))

    return timeline


def _make_timepoint(messages):
    total = len(messages)
    counts = Counter(_ for _ in chain(*(m.words for m in messages)))

    return total, counts


def _load_timeline(paths):
    if os.path.exists(TIMELINE):
        with open(TIMELINE, "rb") as f:
            timeline = pickle.load(f)
    else:
        timeline = _parse_chat(paths)
        with open(TIMELINE, "wb") as f:
            pickle.dump(timeline, f)

    return timeline


def _save_counts(timeline):
    _, _, counters = zip(*timeline)

    counter = Counter()
    for c in counters:
        counter.update(c)

    with open("words.tab", 'w') as f:
        for w, c in sorted(counter.items(), key=itemgetter(1), reverse=True):
            print(w, c, sep='\t', file=f)


def _plot(timeline):
    for npage in range(1, 1 + PAGES):
        chunklen = int(len(timeline) / PAGES / ROW)

        fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR
        plt.subplots_adjust(left=0.07, bottom=0.05, top=0.92)

        for i in range(1, 1 + ROW):
            nrow = i + ROW * (npage - 1)
            f, t = chunklen * (nrow - 1), chunklen * nrow
            x, c, y = zip(*timeline[f:t])
            _x = tuple(t.replace(tzinfo=None) for t in x)

            ax = fig.add_subplot(ROW, 1, i)
            _plot_row(ax, _x, y, c, i == 1, i == ROW)

        fig.suptitle(f"RTA in Japan Summer 2021 チャット頻出スタンプ・単語 ({npage}/{PAGES})",
                     color=FONT_COLOR, size="x-large")
        fig.text(0.03, 0.5, "単語 / 分 （同一メッセージ内の重複は除外）",
                 ha="center", va="center", rotation="vertical", color=FONT_COLOR, size="large")
        fig.savefig(f"{npage}.png", dpi=DPI)
        plt.close()


def moving_average(x, w=AVR_WINDOW):
    return np.convolve(x, np.ones(w), "same") / w


def _plot_row(ax, x, y, total, add_upper_legend, add_lower_legend):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=TIMEZONE))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(range(0, 60, 5)))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.set_facecolor(FACE_COLOR)
    for axis in ("top", "bottom", "left", "right"):
        ax.spines[axis].set_color(FRAME_COLOR)

    ax.tick_params(colors=FONT_COLOR, which="both")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, YMAX)

    total = moving_average(total) * PER_SECONDS
    total = ax.plot(x, total, color=BACKGROUND_COLOR)

    for game in GAMES:
        if x[0] <= game.startat <= x[-1]:
            ax.axvline(x=game.startat, color=ARROW_COLOR, linestyle=":")
            ax.annotate(game.name, xy=(game.startat, YMAX), xytext=(game.startat, YMAX * 0.85), verticalalignment="top",
                        color=FONT_COLOR, arrowprops=dict(facecolor=ARROW_COLOR, shrink=0.05), ha=game.align)

    for words, style, color in RTA_EMOTES:
        if isinstance(words, str):
            words = (words, )
        _y = np.fromiter((sum(c[w] for w in words) for c in y), int)
        if not sum(_y):
            continue
        _y = moving_average(_y) * PER_SECONDS
        ax.plot(x, _y, label="\n".join(words), linestyle=style, color=(color if color else None))

    if add_upper_legend:
        leg = ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        _set_legend(leg)

    if add_lower_legend:
        leg = ax.legend([total[0]], ["メッセージ / 分"], bbox_to_anchor=(1.01, 1), loc="upper left")
        _set_legend(leg)


def _set_legend(leg):
    frame = leg.get_frame()
    frame.set_facecolor(FACE_COLOR)
    frame.set_edgecolor(FRAME_COLOR)

    for text in leg.get_texts():
        text.set_color(FONT_COLOR)


def _main():
    timeline = _load_timeline(sys.argv[1:])
    _save_counts(timeline)
    _plot(timeline)


if __name__ == "__main__":
    _main()
