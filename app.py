import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


WHATSAPP_PATTERNS = [
    # Android-style, e.g. 22/10/2023, 11:01 pm - Name: Message
    re.compile(
        r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2})(?:\s*(?P<ampm>[ap]m))?\s*[-\u2013\u2014]\s*(?P<sender>[^:]+):\s*(?P<message>[\s\S]+)$",
        flags=re.IGNORECASE | re.MULTILINE,
    ),
    # iOS-style, e.g. [22/10/2023, 23:01] Name: Message
    re.compile(
        r"^\[?(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2})(?:\s*(?P<ampm>[ap]m))?\]?\s*(?P<sender>[^:]+):\s*(?P<message>[\s\S]+)$",
        flags=re.IGNORECASE | re.MULTILINE,
    ),    # System messages without sender, e.g. 22/10/2023, 11:01 pm - System message text
    re.compile(
        r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2})(?:\s*(?P<ampm>[ap]m))?\s*[-\u2013\u2014]\s*(?P<message>(?!.+:)[\s\S]+)$",
        flags=re.IGNORECASE | re.MULTILINE,
    ),]

SYSTEM_MESSAGE_SNIPPETS = [
    # Only exact matches of these full messages will be treated as system messages/outliers
    "Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them. Learn more.",
    "You updated the message timer. New messages will disappear from this chat 24 hours after they're sent, except when kept. Tap to change.",
    "You turned off disappearing messages. Tap to change.",
    "You deleted this message",
    "This message was deleted",
    "Messages to this chat are now secured",
    "Messages you send to this group are now secured",
    "You changed the subject",
    "You changed the group icon",
    "You changed this group's description",
    "You created group",
    "You added",
    "You removed",
    "You left",
    "You joined",
    "is a contact",
    "You missed a voice call",
    "You missed a video call",
]

MEDIA_MESSAGE_SNIPPETS = [
    "<Media omitted>",
    "(file attached)",
    "IMG-",
    "VID-",
    "AUD-",
    "PTT-",
    "STK-",
    "DOC-",
]

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "to",
    "is",
    "are",
    "am",
    "i",
    "you",
    "me",
    "we",
    "they",
    "he",
    "she",
    "it",
    "for",
    "on",
    "at",
    "this",
    "that",
    "with",
    "be",
    "was",
    "were",
    "will",
    "can",
    "just",
    "so",
}


def parse_datetime(date_str: str, time_str: str, ampm: str | None) -> datetime:
    """Parse WhatsApp date/time supporting both D/M/Y and M/D/Y, 12h and 24h formats."""
    # Many regions export as DD/MM/YYYY, others as MM/DD/YYYY. Try both.
    date_formats = ["%d/%m/%y", "%d/%m/%Y", "%m/%d/%y", "%m/%d/%Y"]
    time_part = f"{time_str} {ampm or ''}".strip()
    time_format = "%I:%M %p" if ampm else "%H:%M"

    for d_fmt in date_formats:
        fmt = f"{d_fmt} {time_format}"
        try:
            return datetime.strptime(f"{date_str} {time_part}", fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime: {date_str} {time_str} {ampm or ''}")


def format_date_ddmmyy(dt: datetime) -> str:
    """Format datetime to DD/MM/YY format."""
    return dt.strftime("%d/%m/%y")


def extract_emojis(text: str) -> List[str]:
    """
    Extract emoji "grapheme clusters" from text.

    Rules implemented (per requirements):
    - Emoji-only messages are counted (each emoji cluster = 1)
    - Text+emoji messages are counted the same way
    - Multiple emojis in a single message are all counted
    - Repeated emojis are counted every time they appear
    - Skin-tone modifiers and ZWJ sequences (family / handshake etc.) count as ONE emoji
    """
    # Base emoji ranges (single codepoints)
    base = (
        "["
        "\U0001F300-\U0001F5FF"  # Misc symbols & pictographs
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F680-\U0001F6FF"  # Transport & map
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002700-\U000027BF"  # Dingbats
        "]"
    )
    # Optional variation selector or skin-tone modifier
    modifier = "[\uFE0F\U0001F3FB-\U0001F3FF]?"
    zwj = "\u200D"

    # One emoji cluster: base + optional modifier, optionally chained with ZWJ to other clusters
    emoji_cluster = f"(?:{base}{modifier}(?:{zwj}{base}{modifier})*)"
    emoji_pattern = re.compile(emoji_cluster)
    return emoji_pattern.findall(text)


def is_system_message(content: str) -> bool:
    """
    Treat only exact matches from SYSTEM_MESSAGE_SNIPPETS as system messages.
    This relies on the original WhatsApp export text being preserved.
    """
    return content.strip() in SYSTEM_MESSAGE_SNIPPETS


def is_media_message(content: str) -> bool:
    lowered = content.lower()
    return any(snippet.lower() in lowered for snippet in MEDIA_MESSAGE_SNIPPETS)


def is_emoji_only_message(content: str) -> bool:
    """Check if message contains only emojis (and optional whitespace)."""
    # Remove all emoji-related characters comprehensively:
    # - Base emoji ranges
    # - Skin tone modifiers (U+1F3FB to U+1F3FF)
    # - ZWJ (U+200D)
    # - Variation selector (U+FE0F)
    # - Gender modifiers and other combining characters (U+2640, U+2642, U+2695)
    # - Whitespace
    emoji_pattern = r"[\s\u200D\uFE0F\u2640\u2642\u2695\U0001F3FB-\U0001F3FF" \
                    r"\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF" \
                    r"\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF" \
                    r"\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002700-\U000027BF]"
    
    remaining = re.sub(emoji_pattern, "", content)
    
    # If nothing remains and original had content, it's emoji-only
    return len(content.strip()) > 0 and len(remaining) == 0


def parse_chat_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse raw chat lines into structured records."""
    messages = []
    current_msg = None
    
    for raw in lines:
        line = raw.rstrip("\n\r")  # Keep leading/trailing spaces, just remove newlines
        
        # Try to match a message header (date/time - sender: message OR date/time - message)
        matched = None
        for pattern in WHATSAPP_PATTERNS:
            matched = pattern.match(line)
            if matched:
                break
        
        if matched:
            # Save previous message if it exists
            if current_msg and current_msg["message"].strip():
                messages.append(current_msg)
            
            # Start a new message
            groups = matched.groupdict()
            try:
                timestamp = parse_datetime(groups["date"], groups["time"], groups.get("ampm"))
            except ValueError:
                current_msg = None
                continue

            sender = (groups.get("sender") or "").strip() or "System"
            message = groups["message"].strip()
            
            current_msg = {
                "timestamp": timestamp,
                "sender": sender,
                "message": message,
            }
        else:
            # This is a continuation of the previous message
            if current_msg is not None:
                current_msg["message"] += " " + line.strip()
    
    # Don't forget the last message
    if current_msg and current_msg["message"].strip():
        messages.append(current_msg)
    
    # Now process outlier flags
    for msg in messages:
        sys_flag = is_system_message(msg["message"])
        media_flag = is_media_message(msg["message"])
        emoji_only_flag = is_emoji_only_message(msg["message"])
        is_outlier = sys_flag or media_flag  # Only system and media are outliers
        
        msg["is_system"] = sys_flag
        msg["is_outlier"] = is_outlier
        msg["is_emoji_only"] = emoji_only_flag
    
    return messages


def feature_engineer(messages: List[Dict[str, Any]], you_name: str | None) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
    """Create numerical features and enriched message records."""
    if not messages:
        return np.array([]), [], you_name or "You"

    inferred_you = you_name
    if not inferred_you:
        # Heuristic: choose sender with "you" token else the first sender
        senders = Counter([m["sender"] for m in messages])
        inferred_you = next((s for s in senders if s.lower() == "you"), list(senders.keys())[0])

    enriched = []
    features = []
    prev_ts = None
    for msg in messages:
        ts = msg["timestamp"]
        gap_minutes = (ts - prev_ts).total_seconds() / 60 if prev_ts else 0.0
        prev_ts = ts

        words = re.findall(r"\b\w+\b", msg["message"])
        emojis = extract_emojis(msg["message"])

        hour = ts.hour
        word_count = len(words)
        emoji_count = len(emojis)
        sender_flag = 0 if msg["sender"] == inferred_you else 1
        late_night = 1 if 0 <= hour <= 5 else 0
        quick_reply = 1 if gap_minutes <= 5 and gap_minutes > 0 else 0

        feature_vec = [hour, gap_minutes, word_count, emoji_count, sender_flag, late_night, quick_reply]
        features.append(feature_vec)
        enriched.append(
            {
                **msg,
                "gap_minutes": gap_minutes,
                "word_count": word_count,
                "emoji_count": emoji_count,
                "sender_flag": sender_flag,
                "late_night": late_night,
                "quick_reply": quick_reply,
            }
        )

    return np.array(features, dtype=float), enriched, inferred_you


def choose_eps(features: np.ndarray) -> float:
    """Derive an eps value using neighbor distances for DBSCAN."""
    n = len(features)
    if n < 2:
        return 0.5
    k = min(5, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(features)
    distances, _ = nbrs.kneighbors(features)
    # Use median distance to the kth neighbor as baseline
    base = np.median(distances[:, -1])
    return float(max(base * 1.5, 0.5))


def run_dbscan(features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.array([])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    eps = choose_eps(scaled)
    model = DBSCAN(eps=eps, min_samples=3)
    return model.fit_predict(scaled)


def aggregate_stats(enriched: List[Dict[str, Any]], labels: np.ndarray, you_name: str, friend_name: str | None = None) -> Dict[str, Any]:
    if not enriched:
        return {}

    # Attach labels
    for msg, label in zip(enriched, labels if len(labels) else [-1] * len(enriched)):
        msg["cluster"] = int(label)

    # For message count: count all non-outlier, non-emoji-only messages
    # Outliers are: system messages and media messages
    count_msgs = [m for m in enriched if not m.get("is_outlier") and not m.get("is_emoji_only")]
    
    # For behavioral analysis: use clustered messages (exclude DBSCAN noise) that are also non-outliers
    clustered_msgs = [m for m in enriched if m["cluster"] != -1] or enriched
    analysis_msgs = [m for m in clustered_msgs if not m.get("is_outlier")] or clustered_msgs

    def by_sender(msgs_list, filter_fn=None):
        you_count = 0
        other_count = 0
        for m in msgs_list:
            if filter_fn and not filter_fn(m):
                continue
            if m["sender"] == you_name:
                you_count += 1
            else:
                other_count += 1
        return {"you": you_count, "other": other_count, "total": you_count + other_count}

    total_messages = by_sender(count_msgs)
    late_night = by_sender(analysis_msgs, lambda m: m["late_night"] == 1)
    quick_replies = by_sender(analysis_msgs, lambda m: m["quick_reply"] == 1)

    # Calculate chat timeline (start and end dates) - exclude system messages and messages without sender name
    non_system = [m for m in enriched if not m.get("is_system") and m["sender"] != "System"]
    if non_system:
        chat_start = min(m["timestamp"] for m in non_system).strftime("%d-%m-%Y")
        chat_end = max(m["timestamp"] for m in non_system).strftime("%d-%m-%Y")
        unique_dates = set(m["timestamp"].strftime("%Y-%m-%d") for m in non_system)
        total_chat_days = len(unique_dates)
    else:
        chat_start = None
        chat_end = None
        total_chat_days = 0

    emoji_counts = Counter()
    emoji_counts_you = Counter()
    for m in enriched:
        emojis = extract_emojis(m["message"])
        emoji_counts.update(emojis)
        if m["sender"] == you_name:
            emoji_counts_you.update(emojis)

    word_counts_you = Counter()
    word_counts_other = Counter()
    for m in analysis_msgs:
        words = [w.lower() for w in re.findall(r"\b\w+\b", m["message"]) if w.lower() not in STOPWORDS]
        if m["sender"] == you_name:
            word_counts_you.update(words)
        else:
            word_counts_other.update(words)

    activity = Counter()
    activity_with_ts = {}
    for m in analysis_msgs:
        date_str = format_date_ddmmyy(m["timestamp"])
        activity[date_str] += 1
        if date_str not in activity_with_ts:
            activity_with_ts[date_str] = m["timestamp"]

    # Timeline graph data: include all non-system messages (text, emoji-only, media) but exclude system messages
    timeline_activity = Counter()
    timeline_activity_with_ts = {}
    for m in enriched:
        if not m.get("is_system"):  # Exclude system messages only
            date_str = format_date_ddmmyy(m["timestamp"])
            timeline_activity[date_str] += 1
            if date_str not in timeline_activity_with_ts:
                timeline_activity_with_ts[date_str] = m["timestamp"]

    cluster_summary = defaultdict(lambda: {"count": 0, "you": 0, "other": 0, "late_night": 0, "quick": 0})
    for m in enriched:
        c = cluster_summary[m["cluster"]]
        c["count"] += 1
        if m["sender"] == you_name:
            c["you"] += 1
        else:
            c["other"] += 1
        if m["late_night"]:
            c["late_night"] += 1
        if m["quick_reply"]:
            c["quick"] += 1

    scatter_points = [
        {"time_gap": m["gap_minutes"], "emoji_count": m["emoji_count"], "label": int(m["cluster"])} for m in enriched
    ]

    # Sort activity by actual timestamp, then format as DD/MM/YY
    sorted_activity = sorted(activity_with_ts.items(), key=lambda x: x[1])
    messages_over_time = [{"date": d, "count": activity[d]} for d, _ in sorted_activity]

    # Timeline graph data: all non-system messages sorted by date
    sorted_timeline = sorted(timeline_activity_with_ts.items(), key=lambda x: x[1])
    timeline_data = [{"date": d, "count": timeline_activity[d]} for d, _ in sorted_timeline]

    outlier_count = len([m for m in enriched if m.get("is_outlier") and not m.get("is_emoji_only")])

    return {
        "you_name": you_name,
        "friend_name": friend_name,
        "chat_start": chat_start,
        "chat_end": chat_end,
        "total_messages": total_messages,
        "total_emojis": {"total": sum(emoji_counts.values()), "you": sum(emoji_counts_you.values()), "other": sum(emoji_counts.values()) - sum(emoji_counts_you.values())},
        "top_words": {
            "you": word_counts_you.most_common(5),
            "other": word_counts_other.most_common(5),
        },
        "top_emojis": emoji_counts.most_common(5),
        "late_night": late_night,
        "quick_replies": quick_replies,
        "messages_over_time": messages_over_time,
        "timeline_data": timeline_data,
        "cluster_summary": [{"label": int(lbl), **vals} for lbl, vals in sorted(cluster_summary.items(), key=lambda x: x[0])],
        "scatter_points": scatter_points,
        "labels": list(set(m["cluster"] for m in enriched)),
        "noise_count": len([m for m in enriched if m["cluster"] == -1]),
        "content_outlier_count": outlier_count,
    }


def analyze_chat(file_text: str, you_name: str | None, friend_name: str | None = None) -> Dict[str, Any]:
    lines = file_text.splitlines()
    parsed = parse_chat_lines(lines)
    # If both names are provided, restrict to this two-person conversation
    # BUT keep system/outlier messages as they're important for context
    if you_name and friend_name:
        allowed = {you_name.strip(), friend_name.strip()}
        parsed = [m for m in parsed if m["sender"] in allowed or m.get("is_outlier")]
    features, enriched, inferred_you = feature_engineer(parsed, you_name)
    labels = run_dbscan(features)
    stats = aggregate_stats(enriched, labels, inferred_you, friend_name)
    stats["total_messages_raw"] = len(parsed)
    return stats


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    you_name = request.form.get("you_name") or None
    friend_name = request.form.get("friend_name") or None
    if not file:
        return jsonify({"error": "No file provided"}), 400
    if not file.filename.endswith(".txt"):
        return jsonify({"error": "Only .txt exports are supported"}), 400
    try:
        content = file.read().decode("utf-8")
    except UnicodeDecodeError:
        content = file.read().decode("utf-16")
    try:
        result = analyze_chat(content, you_name, friend_name)
        return jsonify(result)
    except Exception as exc:  # pragma: no cover - API safeguard
        return jsonify({"error": f"Failed to analyze chat: {exc}"}), 500


@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    data = request.get_json(silent=True) or {}
    you_name = data.get("you_name")
    friend_name = data.get("friend_name")
    content = data.get("content")
    if not content:
        return jsonify({"error": "Missing content"}), 400
    result = analyze_chat(content, you_name, friend_name)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)

