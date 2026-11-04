"""Auto-emoji for captions: the look OpusClip/Submagic popularised — when a
keyword is spoken, its emoji rides next to the caption. Pure keyword map,
fully offline, no model."""

from __future__ import annotations

import re
from typing import List, Sequence

EMOJI = {
    # money & work
    "money": "💰", "soldi": "💰", "cash": "💵", "price": "💲", "prezzo": "💲",
    "free": "🆓", "gratis": "🆓", "business": "💼", "lavoro": "💼", "work": "💼",
    "sales": "📈", "vendite": "📈", "investimento": "📈", "investment": "📈",
    "marketing": "📣", "brand": "™️", "cliente": "🤝", "client": "🤝",
    # fire / amazing
    "fire": "🔥", "fuoco": "🔥", "amazing": "😲", "incredible": "🤯",
    "incredibile": "🤯", "pazzesco": "🤯", "crazy": "🤪", "insane": "🤯",
    "wow": "😮", "top": "🔝", "best": "🏆", "migliore": "🏆", "winner": "🏆",
    "vincitore": "🏆", "win": "🏅", "vinto": "🏅",
    # love & feelings
    "love": "❤️", "amore": "❤️", "cuore": "❤️", "heart": "❤️",
    "odio": "😤", "hate": "😤", "fear": "😱", "paura": "😱", "scary": "😨",
    "happy": "😊", "felice": "😊", "sad": "😢", "triste": "😢",
    "angry": "😡", "arrabbiato": "😡", "sorry": "😅", "scusa": "😅",
    # ideas & secrets
    "idea": "💡", "secret": "🤫", "segreto": "🤫", "truth": "🎯", "verità": "🎯",
    "question": "❓", "domanda": "❓", "answer": "✅", "risposta": "✅",
    "problem": "⚠️", "problema": "⚠️", "errore": "❌", "mistake": "❌",
    "tip": "💡", "trick": "🪄", "trucco": "🪄", "hack": "🛠️",
    # people & body
    "eyes": "👀", "occhi": "👀", "look": "👀", "guarda": "👀", "watch": "👀",
    "brain": "🧠", "cervello": "🧠", "mind": "🧠", "mente": "🧠",
    "man": "👨", "woman": "👩", "uomo": "👨", "donna": "👩", "people": "👥",
    "gente": "👥", "friend": "🫂", "amico": "🫂", "amica": "🫂",
    "kids": "🧒", "bambini": "🧒", "family": "👪", "famiglia": "👪",
    # time & speed
    "time": "⏰", "tempo": "⏰", "fast": "⚡", "veloce": "⚡", "slow": "🐢",
    "lento": "🐢", "now": "⏳", "subito": "⏳", "adesso": "⏳",
    "never": "🚫", "mai": "🚫", "always": "♾️", "sempre": "♾️",
    "today": "📅", "oggi": "📅", "tomorrow": "🔮", "domani": "🔮",
    "yesterday": "⏪", "ieri": "⏪", "future": "🔮", "futuro": "🔮",
    # tech & media
    "video": "🎬", "film": "🎬", "movie": "🎬", "music": "🎵", "musica": "🎵",
    "song": "🎤", "canzone": "🎤", "photo": "📸", "foto": "📸",
    "phone": "📱", "telefono": "📱", "computer": "💻", "internet": "🌐",
    "ai": "🤖", "robot": "🤖", "game": "🎮", "gioco": "🎮", "play": "▶️",
    # places & things
    "house": "🏠", "casa": "🏠", "car": "🚗", "macchina": "🚗", "auto": "🚗",
    "food": "🍕", "cibo": "🍕", "pizza": "🍕", "coffee": "☕", "caffè": "☕",
    "wine": "🍷", "vino": "🍷", "beer": "🍺", "birra": "🍺",
    "book": "📚", "libro": "📚", "school": "🏫", "scuola": "🏫",
    "gym": "💪", "palestra": "💪", "strong": "💪", "forte": "💪",
    "run": "🏃", "corsa": "🏃", "travel": "✈️", "viaggio": "✈️",
    "sea": "🌊", "mare": "🌊", "sun": "☀️", "sole": "☀️", "rain": "🌧️",
    "pioggia": "🌧️", "night": "🌙", "notte": "🌙", "day": "🌞", "giorno": "🌞",
    # reactions
    "yes": "✅", "sì": "✅", "no": "❌", "stop": "🛑", "ferma": "🛑",
    "go": "🚀", "vai": "🚀", "start": "🎬", "inizia": "🎬", "end": "🏁",
    "fine": "🏁", "finish": "🏁", "new": "🆕", "nuovo": "🆕",
    "big": "🐘", "grande": "🐘", "small": "🐜", "piccolo": "🐜",
    "god": "🙏", "dio": "🙏", "luck": "🍀", "fortuna": "🍀",
    "dream": "💭", "sogno": "💭", "goal": "🎯", "obiettivo": "🎯",
    "party": "🎉", "festa": "🎉", "birthday": "🎂", "compleanno": "🎂",
    "gift": "🎁", "regalo": "🎁", "king": "👑", "re": "👑", "queen": "👑",
    "regina": "👑", "star": "⭐", "stella": "⭐", "diamond": "💎",
    "diamante": "💎", "gold": "🥇", "oro": "🥇",
}

_NORM = re.compile(r"[^\wàèéìòù'’]+")


def for_words(texts: Sequence[str], limit: int = 2) -> List[str]:
    """Emoji for the keywords found in ``texts``, in spoken order (deduped)."""
    out: List[str] = []
    for t in texts:
        for tok in t.split():
            e = EMOJI.get(_NORM.sub("", tok.lower()))
            if e and e not in out:
                out.append(e)
                if len(out) >= limit:
                    return out
    return out
