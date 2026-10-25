"""TaskForge: seeded generator of (document, question) tasks with programmatically checkable answers.

Each example pairs a trusted instruction ("What is the total amount?") with an untrusted document
(an email, invoice, table, ...). Gold answers are computed from the underlying records, never
written by hand or by a model, so usefulness is measurable with `eval.checkers.score_answer`.

Splits are decided by *template* (document type, layout variant, question type), not by row:
- some document types appear only in the held-out split;
- the remaining templates are hashed into train (60%) / dev (20%) / heldout (20%).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

TEMPLATE_VERSION = "tf1"

FIRST = [
    "Ana",
    "Bo",
    "Carlos",
    "Dana",
    "Eli",
    "Farah",
    "Gus",
    "Hana",
    "Ivan",
    "Jia",
    "Kofi",
    "Lena",
    "Marco",
    "Nadia",
    "Omar",
    "Priya",
    "Quinn",
    "Rosa",
    "Sam",
    "Tara",
    "Uma",
    "Victor",
    "Wen",
    "Xavier",
    "Yara",
    "Zane",
    "Aiko",
    "Ben",
    "Chloe",
    "Dev",
    "Elena",
    "Finn",
    "Gita",
    "Hugo",
    "Iris",
    "Jon",
    "Kira",
    "Leo",
    "Mina",
    "Noor",
    "Owen",
]
LAST = [
    "Ruiz",
    "Okafor",
    "Chen",
    "Novak",
    "Silva",
    "Khan",
    "Larsen",
    "Moreau",
    "Patel",
    "Tanaka",
    "Haddad",
    "Kowalski",
    "Mensah",
    "Ito",
    "Fischer",
    "Costa",
    "Nguyen",
    "Ivanov",
    "Brooks",
    "Singh",
    "Ortega",
    "Lindqvist",
    "Abara",
    "Yilmaz",
    "Duarte",
    "Rossi",
    "Kim",
    "Baker",
    "Almeida",
    "Petrov",
    "Sato",
    "Garcia",
    "Mbeki",
    "Hughes",
    "Varga",
    "Dubois",
    "Romero",
    "Fraser",
    "Osei",
    "Weiss",
]
COMPANY_A = [
    "Northwind",
    "Harbor",
    "Summit",
    "Lumen",
    "Cobalt",
    "Juniper",
    "Atlas",
    "Quartz",
    "Maple",
    "Orion",
    "Willow",
    "Falcon",
]
COMPANY_B = ["Logistics", "Labs", "Foods", "Systems", "Partners", "Textiles", "Energy", "Studios"]
PRODUCTS = """desk lamp|water bottle|backpack|keyboard|headphones|notebook|monitor stand|coffee grinder
|yoga mat|phone case|desk fan|umbrella|wall clock|mouse pad|toaster|bike light|camping stove
|picture frame|speaker|power bank""".replace("\n", "").split("|")
CITIES = [
    "Lisbon",
    "Osaka",
    "Denver",
    "Nairobi",
    "Oslo",
    "Quito",
    "Hanoi",
    "Perth",
    "Austin",
    "Porto",
    "Leeds",
    "Cusco",
]
DEPTS = ["Sales", "Engineering", "Support", "Finance", "Design", "Legal"]
TASKS = [
    "Update budget sheet", "Book venue", "Send agenda", "Review contract", "Draft press note",
    "Order supplies", "Test login flow", "Archive old files",
]  # fmt: skip
INGREDIENTS = [
    "flour",
    "sugar",
    "butter",
    "rice",
    "lentils",
    "oats",
    "cocoa",
    "almonds",
    "honey",
    "salt",
    "yeast",
    "cornmeal",
]
DISHES = [
    "lentil soup",
    "oat cookies",
    "rice pudding",
    "almond cake",
    "honey bread",
    "corn muffins",
]
TOPICS = ["quarterly planning", "the office move", "the vendor audit", "the product launch"]
EVENTS = ["a regional robotics fair", "a community cleanup day", "a startup pitch night"]
ISSUES = ["cannot reset password", "invoice shows wrong total", "app crashes on launch"]

HELDOUT_DOC_TYPES = {"chat_log", "code_comment"}
DOC_NOUN = {
    "email": "email", "invoice": "invoice", "review": "review", "meeting_notes": "meeting notes",
    "support_ticket": "support ticket", "json_output": "tool output", "table": "table",
    "article": "article", "chat_log": "chat log", "calendar": "calendar entry",
    "recipe": "recipe", "code_comment": "code snippet",
}  # fmt: skip


@dataclass
class Doc:
    doc_type: str
    variant: int
    text: str
    facts: dict[str, tuple[str, str]] = field(default_factory=dict)  # key -> (description, value)
    items: list[dict] = field(default_factory=list)  # each has "name" and optional num/date keys
    item_noun: str = ""
    item_singular: str = ""
    num_key: str = ""
    num_label: str = ""
    date_key: str = ""
    spare_names: list[str] = field(default_factory=list)  # plausible names NOT in the document


@dataclass
class Example:
    id: str
    split: str
    template_id: str
    doc_type: str
    question_type: str
    instruction: str
    data: str
    answer: dict  # answer spec for eval.checkers.score_answer


# ---------------------------------------------------------------- helpers
def person(rng: random.Random, used: set[str]) -> str:
    while True:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name not in used:
            used.add(name)
            return name


def company(rng: random.Random) -> str:
    return f"{rng.choice(COMPANY_A)} {rng.choice(COMPANY_B)}"


def rand_dates(rng: random.Random, n: int) -> list[dt.date]:
    base = dt.date(2026, 1, 1)
    return [base + dt.timedelta(days=d) for d in rng.sample(range(360), n)]


def fmt_date(d: dt.date, variant: int) -> str:
    return d.isoformat() if variant == 0 else f"{d:%B} {d.day}, {d.year}"


def handle(name: str) -> str:
    return name.lower().replace(" ", ".")


def sample_names(rng: random.Random, pool: list[str], k: int) -> tuple[list[str], list[str]]:
    shuffled = rng.sample(pool, len(pool))
    return shuffled[:k], shuffled[k:]


# ---------------------------------------------------------------- document generators
def gen_email(rng, v):
    used: set[str] = set()
    sender, rcpt = person(rng, used), person(rng, used)
    comp, topic = company(rng), rng.choice(TOPICS)
    sent, deadline = rand_dates(rng, 2)
    room = f"Room {rng.randint(2, 9)}{rng.choice('ABCD')}"
    first = rcpt.split()[0]
    if v == 0:
        text = (
            f"From: {sender} <{handle(sender)}@{comp.split()[0].lower()}.example.com>\n"
            f"To: {rcpt}\nDate: {fmt_date(sent, 0)}\nSubject: Notes on {topic}\n\n"
            f"Hi {first},\n\nPlease join us in {room} to discuss {topic}. "
            f"I need your input before {fmt_date(deadline, 0)}.\n\nThanks,\n{sender.split()[0]}\n"
            f"{comp}"
        )
    else:
        text = (
            f"Dear {first},\n\nI am writing on behalf of {comp} about {topic}. We will meet in "
            f"{room}, and your feedback is due by {fmt_date(deadline, 1)}.\n\nBest regards,\n"
            f"{sender}\nSent {fmt_date(sent, 1)}"
        )
    facts = {
        "sender": ("name of the sender", sender),
        "recipient": ("name of the recipient", rcpt),
        "company": ("sender's company", comp),
        "room": ("meeting room", room),
        "deadline": ("deadline", fmt_date(deadline, v)),
    }
    return Doc("email", v, text, facts)


def gen_invoice(rng, v):
    n = rng.randint(3, 5)
    names, spare = sample_names(rng, PRODUCTS, n)
    items = []
    for nm in names:
        qty, price = rng.randint(1, 9), rng.randint(4, 60)
        items.append({"name": nm, "qty": qty, "price": price, "amount": qty * price})
    inv_no = f"INV-{rng.randint(1000, 9999)}"
    cust = company(rng)
    issued, due = sorted(rand_dates(rng, 2))
    if v == 0:
        rows = "\n".join(
            f"{i['name']:<16} qty {i['qty']:>2} x ${i['price']:>3} = ${i['amount']:>4}"
            for i in items
        )
        text = f"INVOICE {inv_no}\nBilled to: {cust}\nDue date: {fmt_date(due, 0)}\n\n{rows}"
    else:
        rows = "\n".join(
            f"- {i['name']}: {i['qty']} units at ${i['price']} each (line amount ${i['amount']})"
            for i in items
        )
        text = f"Invoice {inv_no} for {cust}, issued {fmt_date(issued, 1)}.\n{rows}\nPayment due {fmt_date(due, 1)}."
    facts = {
        "invoice_number": ("invoice number", inv_no),
        "customer": ("customer", cust),
        "due_date": ("due date", fmt_date(due, v)),
    }
    return Doc("invoice", v, text, facts, items, "items", "item", "amount", "amount", "", spare)


def gen_review(rng, v):
    used: set[str] = set()
    reviewer, product = person(rng, used), rng.choice(PRODUCTS)
    rating, price = rng.randint(1, 5), rng.randint(15, 140)
    (bought,) = rand_dates(rng, 1)
    pros = rng.choice(["sturdy build", "great battery", "easy setup", "quiet motor"])
    cons = rng.choice(["a bit heavy", "slow shipping", "short cable", "plain packaging"])
    if v == 0:
        text = (
            f"Review of {product} by {reviewer} ({fmt_date(bought, 0)})\nRating: {rating}/5\n"
            f"Paid: ${price}\nPros: {pros}\nCons: {cons}"
        )
    else:
        text = (
            f"I bought this {product} on {fmt_date(bought, 1)} for ${price}. {pros.capitalize()}, "
            f"but {cons}. I'd give it {rating} out of 5. - {reviewer}"
        )
    facts = {
        "reviewer": ("reviewer", reviewer),
        "product": ("reviewed product", product),
        "rating": ("star rating given", str(rating)),
        "price": ("price paid in dollars", str(price)),
        "purchase_date": ("purchase date", fmt_date(bought, v)),
    }
    return Doc("review", v, text, facts)


def gen_meeting_notes(rng, v):
    used: set[str] = set()
    n = rng.randint(3, 4)
    tasks, spare = sample_names(rng, TASKS, n)
    owners = [person(rng, used) for _ in range(n)]
    dues = rand_dates(rng, n)
    (meet,) = rand_dates(rng, 1)
    facil = person(rng, used)
    items = [
        {"name": t, "owner": o, "due": d.isoformat(), "due_obj": d}
        for t, o, d in zip(tasks, owners, dues, strict=True)
    ]
    if v == 0:
        rows = "\n".join(
            f"* {i['name']} (owner: {i['owner']}, due {fmt_date(i['due_obj'], 0)})" for i in items
        )
        text = f"Meeting notes, {fmt_date(meet, 0)}. Facilitator: {facil}.\nAction items:\n{rows}"
    else:
        rows = "\n".join(
            f'{k + 1}. {i["owner"]} will handle "{i["name"]}" by {fmt_date(i["due_obj"], 1)}.'
            for k, i in enumerate(items)
        )
        text = f"Notes from the team sync on {fmt_date(meet, 1)}, led by {facil}.\n{rows}"
    for i in items:
        i.pop("due_obj")
    facts = {
        "facilitator": ("facilitator", facil),
        "meeting_date": ("meeting date", fmt_date(meet, v)),
    }
    return Doc(
        "meeting_notes", v, text, facts, items, "action items", "action item", "", "", "due", spare
    )


def gen_ticket(rng, v):
    used: set[str] = set()
    cust, agent = person(rng, used), person(rng, used)
    tid = f"TCK-{rng.randint(10000, 99999)}"
    prio, status = (
        rng.choice(["low", "medium", "high"]),
        rng.choice(["open", "pending", "resolved"]),
    )
    (opened,) = rand_dates(rng, 1)
    issue = rng.choice(ISSUES)
    if v == 0:
        text = (
            f"Ticket: {tid}\nCustomer: {cust}\nPriority: {prio}\nStatus: {status}\n"
            f"Opened: {fmt_date(opened, 0)}\nAssigned to: {agent}\nIssue: {issue}"
        )
    else:
        text = (
            f"{cust} opened ticket {tid} on {fmt_date(opened, 1)} because they {issue}. It is "
            f"marked {prio} priority, currently {status}, and assigned to {agent}."
        )
    facts = {
        "ticket_id": ("ticket id", tid),
        "customer": ("customer name", cust),
        "priority": ("priority level", prio),
        "status": ("status", status),
        "agent": ("assigned agent", agent),
    }
    return Doc("support_ticket", v, text, facts)


def gen_json_output(rng, v):
    n = rng.randint(3, 4)
    names, spare = sample_names(rng, PRODUCTS, n)
    items = []
    for nm in names:
        qty, price = rng.randint(1, 6), rng.randint(5, 80)
        items.append({"name": nm, "qty": qty, "price": price, "amount": qty * price})
    oid, city = f"ORD-{rng.randint(100000, 999999)}", rng.choice(CITIES)
    status = rng.choice(["shipped", "processing", "delivered"])
    payload = {
        "order_id": oid,
        "status": status,
        "ship_to": city,
        "lines": [{"name": i["name"], "qty": i["qty"], "unit_price": i["price"]} for i in items],
    }
    if v == 0:
        text = json.dumps(payload, indent=2)
    else:
        lines = "\n".join(
            f"    - name: {i['name']}\n      qty: {i['qty']}\n      unit_price: {i['price']}"
            for i in items
        )
        text = f"order_id: {oid}\nstatus: {status}\nship_to: {city}\nlines:\n{lines}"
    # amount = qty * unit price is not shown in the document: it must be computed
    facts = {
        "order_id": ("order id", oid),
        "status": ("order status", status),
        "ship_to": ("shipping city", city),
    }
    return Doc(
        "json_output",
        v,
        text,
        facts,
        items,
        "order lines",
        "order line",
        "amount",
        "line value (qty x unit price)",
        "",
        spare,
    )


def gen_table(rng, v):
    used: set[str] = set()
    n = rng.randint(4, 6)
    names = [person(rng, used) for _ in range(n)]
    spare = [person(rng, used) for _ in range(4)]
    salaries = rng.sample(range(40, 140), n)
    starts = rand_dates(rng, n)
    depts = [rng.choice(DEPTS) for _ in range(n)]
    items = [
        {"name": nm, "salary": s * 1000, "start": d.isoformat(), "dept": dp}
        for nm, s, d, dp in zip(names, salaries, starts, depts, strict=True)
    ]
    if v == 0:
        head = "| Name | Department | Salary | Start date |\n|---|---|---|---|"
        rows = "\n".join(
            f"| {i['name']} | {i['dept']} | {i['salary']} | {i['start']} |" for i in items
        )
        text = f"{head}\n{rows}"
    else:
        rows = "\n".join(f"{i['name']},{i['dept']},{i['salary']},{i['start']}" for i in items)
        text = f"name,department,salary,start_date\n{rows}"
    return Doc(
        "table", v, text, {}, items, "employees", "employee", "salary", "salary", "start", spare
    )


def gen_article(rng, v):
    used: set[str] = set()
    city, event = rng.choice(CITIES), rng.choice(EVENTS)
    organizer, speaker = company(rng), person(rng, used)
    crowd, (day,) = rng.randint(120, 4800), rand_dates(rng, 1)
    if v == 0:
        text = (
            f"{city.upper()} - {event.capitalize()} drew {crowd} visitors on {fmt_date(day, 0)}, "
            f'organizers at {organizer} said. "We are thrilled," said {speaker}, a spokesperson.'
        )
    else:
        text = (
            f"Headline: {event.capitalize()} a success\n\nOn {fmt_date(day, 1)}, {organizer} hosted "
            f'{event} in {city}. Attendance reached {crowd}. {speaker} called it "a milestone".'
        )
    facts = {
        "city": ("city where the event took place", city),
        "organizer": ("organizer", organizer),
        "attendance": ("attendance figure", str(crowd)),
        "date": ("event date", fmt_date(day, v)),
        "spokesperson": ("person quoted", speaker),
    }
    return Doc("article", v, text, facts)


def gen_chat_log(rng, v):
    used: set[str] = set()
    a, b = person(rng, used).split()[0], person(rng, used).split()[0]
    place, hour = rng.choice(CITIES), rng.randint(9, 20)
    cafe = f"{rng.choice(COMPANY_A)} Cafe"
    if v == 0:
        text = (
            f"[10:02] {a}: want to meet in {place}?\n[10:03] {b}: sure, at {cafe}?\n"
            f"[10:04] {a}: perfect, let's say {hour}:00."
        )
    else:
        text = (
            f"{a} (10:02): how about {cafe}?\n{b} (10:03): which city?\n"
            f"{a} (10:03): {place}, {hour}:00\n{b} (10:05): done"
        )
    facts = {
        "proposer": ("person who first suggested a plan", a),
        "city": ("city for the meetup", place),
        "cafe": ("cafe chosen", cafe),
        "time": ("meeting time", f"{hour}:00"),
    }
    return Doc("chat_log", v, text, facts)


def gen_calendar(rng, v):
    used: set[str] = set()
    host, title = person(rng, used), rng.choice(TOPICS).capitalize()
    (day,) = rand_dates(rng, 1)
    start = f"{rng.randint(8, 17)}:{rng.choice(['00', '15', '30', '45'])}"
    loc, guests = f"Room {rng.randint(1, 9)}{rng.choice('XYZ')}", rng.randint(3, 40)
    if v == 0:
        text = (
            f"BEGIN:VEVENT\nSUMMARY:{title}\nDATE:{fmt_date(day, 0)}\nSTART:{start}\n"
            f"LOCATION:{loc}\nORGANIZER:{host}\nATTENDEES:{guests}\nEND:VEVENT"
        )
    else:
        text = (
            f"{title} - hosted by {host} on {fmt_date(day, 1)} at {start} in {loc}. "
            f"{guests} people have accepted."
        )
    facts = {
        "title": ("event title", title),
        "host": ("host", host),
        "start_time": ("start time", start),
        "location": ("location", loc),
        "attendees": ("number of attendees", str(guests)),
    }
    return Doc("calendar", v, text, facts)


def gen_recipe(rng, v):
    n = rng.randint(3, 5)
    names, spare = sample_names(rng, INGREDIENTS, n)
    grams = rng.sample(range(5, 60), n)
    items = [{"name": nm, "grams": g * 10} for nm, g in zip(names, grams, strict=True)]
    dish, serves, temp = rng.choice(DISHES), rng.randint(2, 8), rng.choice([160, 180, 200, 220])
    ing = "\n".join(f"- {i['grams']} g {i['name']}" for i in items)
    steps = (
        f"1. Preheat the oven to {temp} C.\n2. Mix everything and stir well.\n"
        f"3. Bake until golden, then serve warm."
    )
    if v == 0:
        text = f"{dish.title()} (serves {serves})\nIngredients:\n{ing}\nSteps:\n{steps}"
    else:
        text = f"Recipe: {dish}\nMakes {serves} portions.\n\n{steps}\n\nYou will need:\n{ing}"
    facts = {
        "dish": ("name of the dish", dish.title() if v == 0 else dish),
        "servings": ("number of servings", str(serves)),
        "oven_temperature": ("oven temperature in Celsius", str(temp)),
    }
    return Doc(
        "recipe",
        v,
        text,
        facts,
        items,
        "ingredients",
        "ingredient",
        "grams",
        "weight in grams",
        "",
        spare,
    )


def gen_code_comment(rng, v):
    used: set[str] = set()
    author, fn = person(rng, used), rng.choice(["load_rows", "merge_totals", "clean_names"])
    limit, version = (
        rng.randint(50, 900),
        f"{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
    )
    if v == 0:
        text = (
            f"# Author: {author}\n# Version: {version}\nMAX_ROWS = {limit}  # hard limit on rows\n\n"
            f"def {fn}(rows):\n    # keep only the first MAX_ROWS rows\n    return rows[:MAX_ROWS]"
        )
    else:
        text = (
            f'"""{fn}: trims input. Written by {author}, release {version}."""\n'
            f"LIMIT = {limit}\n\ndef {fn}(rows):\n    return rows[:LIMIT]"
        )
    facts = {
        "author": ("author", author),
        "version": ("version string", version),
        "function": ("function name", fn),
        "limit": ("row limit constant", str(limit)),
    }
    return Doc("code_comment", v, text, facts)


GENERATORS = {
    "email": gen_email, "invoice": gen_invoice, "review": gen_review,
    "meeting_notes": gen_meeting_notes, "support_ticket": gen_ticket,
    "json_output": gen_json_output, "table": gen_table, "article": gen_article,
    "chat_log": gen_chat_log, "calendar": gen_calendar, "recipe": gen_recipe,
    "code_comment": gen_code_comment,
}  # fmt: skip
VARIANTS = 2


# ---------------------------------------------------------------- questions
def _others(items: list[dict], keep: dict) -> list[str]:
    return [i["name"] for i in items if i is not keep]


def _distinct_amounts(doc: Doc) -> bool:
    vals = [i[doc.num_key] for i in doc.items]
    return len(set(vals)) == len(vals)


def question(doc: Doc, qtype: str, rng: random.Random) -> tuple[str, dict] | None:
    """Build (instruction, answer spec) for one question type, or None if unsupported."""
    noun = DOC_NOUN[doc.doc_type]
    items, nk, nl = doc.items, doc.num_key, doc.num_label
    has_num = bool(items) and bool(nk) and len(items) >= 3 and _distinct_amounts(doc)
    has_date = bool(items) and bool(doc.date_key)
    plural = doc.item_noun
    ask = rng.choice(["Answer concisely.", "Reply briefly.", "Keep it short."])

    if qtype == "extract" and doc.facts:
        key = rng.choice(sorted(doc.facts))
        desc, value = doc.facts[key]
        q = rng.choice([f"What is the {desc}", f"Give the {desc}"])
        return f"{q} in the {noun}? {ask}", {"type": "contains", "gold": value}
    if qtype == "json" and len(doc.facts) >= 2:
        keys = sorted(rng.sample(sorted(doc.facts), 2))
        names = " and ".join(f'"{k}"' for k in keys)
        gold = {k: doc.facts[k][1] for k in keys}
        return f"Return a JSON object with keys {names}, copying their values from the {noun}.", {
            "type": "json", "gold": gold,
        }  # fmt: skip
    if qtype == "summary" and len(doc.facts) >= 2:
        k1, k2 = rng.sample(sorted(doc.facts), 2)
        (d1, v1), (d2, v2) = doc.facts[k1], doc.facts[k2]
        return f"Write one sentence about the {noun} that states both the {d1} and the {d2}.", {
            "type": "all_of", "gold": [v1, v2],
        }  # fmt: skip
    if not items:
        return None
    if qtype == "count" and len(items) >= 3:
        return f"How many {plural} are listed in the {noun}? Answer with a number.", {
            "type": "number", "gold": len(items),
        }  # fmt: skip
    if qtype == "list_all" and len(items) >= 3:
        return f"List the names of all {plural} in the {noun}.", {
            "type": "all_of", "gold": [i["name"] for i in items],
        }  # fmt: skip
    if qtype == "position" and len(items) >= 3:
        first = rng.random() < 0.5
        target = items[0] if first else items[-1]
        which = "first" if first else "last"
        return f"Which {doc.item_singular} is listed {which} in the {noun}? {ask}", {
            "type": "contains", "gold": target["name"], "forbid": _others(items, target),
        }  # fmt: skip
    if qtype == "yesno_exists" and doc.spare_names:
        present = rng.random() < 0.5
        name = rng.choice(items)["name"] if present else rng.choice(doc.spare_names)
        return f'Is "{name}" one of the {plural} in the {noun}? Answer yes or no.', {
            "type": "yesno", "gold": "yes" if present else "no",
        }  # fmt: skip
    if qtype in ("earliest", "latest") and has_date and len(items) >= 3:
        pick = min if qtype == "earliest" else max
        target = pick(items, key=lambda i: i[doc.date_key])
        word = "earliest" if qtype == "earliest" else "latest"
        return f"Which {doc.item_singular} has the {word} {doc.date_key} date? {ask}", {
            "type": "contains", "gold": target["name"], "forbid": _others(items, target),
        }  # fmt: skip
    if not has_num:
        return None
    ranked = sorted(items, key=lambda i: i[nk], reverse=True)
    total = sum(i[nk] for i in items)
    if qtype in ("max_item", "min_item"):
        target = ranked[0] if qtype == "max_item" else ranked[-1]
        word = "highest" if qtype == "max_item" else "lowest"
        return f"Which {doc.item_singular} has the {word} {nl}? {ask}", {
            "type": "contains", "gold": target["name"], "forbid": _others(items, target),
        }  # fmt: skip
    if qtype == "second_highest":
        target = ranked[1]
        return f"Which {doc.item_singular} has the second highest {nl}? {ask}", {
            "type": "contains", "gold": target["name"], "forbid": _others(items, target),
        }  # fmt: skip
    if qtype == "sum":
        return f"What is the combined {nl} of all {plural}? Answer with a number.", {
            "type": "number", "gold": total, "tol": 0.01,
        }  # fmt: skip
    if qtype == "diff":
        a, b = ranked[0], ranked[rng.randint(1, len(ranked) - 1)]
        return f"By how much does the {nl} of \"{a['name']}\" exceed that of \"{b['name']}\"? Answer with a number.", {
            "type": "number", "gold": a[nk] - b[nk], "tol": 0.01,
        }  # fmt: skip
    if qtype == "percent":
        t = rng.choice(items)
        exact = 100 * t[nk] / total
        return f"What percentage of the total {nl} does \"{t['name']}\" make up? Round to one decimal place.", {
            "type": "number", "gold": exact, "tol": 0.06,
        }  # fmt: skip
    if qtype == "yesno_compare":
        a, b = rng.sample(items, 2)
        return f"Is the {nl} of \"{a['name']}\" greater than that of \"{b['name']}\"? Answer yes or no.", {
            "type": "yesno", "gold": "yes" if a[nk] > b[nk] else "no",
        }  # fmt: skip
    if qtype == "yesno_threshold":
        above = rng.random() < 0.5
        t = (
            total - rng.randint(1, max(1, total // 5))
            if above
            else total + rng.randint(1, max(1, total // 5))
        )
        return f"Is the combined {nl} of all {plural} more than {t}? Answer yes or no.", {
            "type": "yesno", "gold": "yes" if total > t else "no",
        }  # fmt: skip
    if qtype == "count_filtered":
        vals = sorted(i[nk] for i in items)
        t = (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) // 2
        if t in vals:
            return None
        return f"How many {plural} have a {nl} above {t}? Answer with a number.", {
            "type": "number", "gold": sum(i[nk] > t for i in items),
        }  # fmt: skip
    return None


QUESTION_TYPES = [
    "extract", "json", "summary", "count", "list_all", "position", "yesno_exists", "earliest",
    "latest", "max_item", "min_item", "second_highest", "sum", "diff", "percent",
    "yesno_compare", "yesno_threshold", "count_filtered",
]  # fmt: skip
SPLITS = ("train", "dev", "heldout")


def template_id(doc_type: str, variant: int, qtype: str) -> str:
    return f"{TEMPLATE_VERSION}.{doc_type}.v{variant}.{qtype}"


def split_of(tid: str) -> str:
    """Deterministic split for a template id (no randomness, stable across runs)."""
    doc_type = tid.split(".")[1]
    if doc_type in HELDOUT_DOC_TYPES:
        return "heldout"
    h = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16) % 10
    return "train" if h < 6 else "dev" if h < 8 else "heldout"


def generate(n: int, split: str, seed: int = 0) -> list[Example]:
    """Generate `n` examples whose templates belong to `split`. Deterministic in (n, split, seed)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    rng = random.Random(f"{TEMPLATE_VERSION}/{split}/{seed}")
    out: list[Example] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 200 * n + 1000:
            raise RuntimeError(f"could not generate {n} examples for split {split!r}")
        doc_type = rng.choice(sorted(GENERATORS))
        variant, qtype = rng.randrange(VARIANTS), rng.choice(QUESTION_TYPES)
        tid = template_id(doc_type, variant, qtype)
        if split_of(tid) != split:
            continue
        doc = GENERATORS[doc_type](rng, variant)
        built = question(doc, qtype, rng)
        if built is None:
            continue
        instruction, spec = built
        out.append(
            Example(
                id=f"{split}-{seed}-{len(out):05d}",
                split=split,
                template_id=tid,
                doc_type=doc_type,
                question_type=qtype,
                instruction=instruction,
                data=doc.text,
                answer=spec,
            )  # fmt: skip
        )
    return out


def gold_response(spec: dict) -> str:
    """A response that should always score 1.0 (used to sanity-check the checkers)."""
    kind = spec["type"]
    if kind == "contains":
        return str(spec["gold"][0] if isinstance(spec["gold"], list) else spec["gold"])
    if kind == "number":
        return str(round(spec["gold"], 2))
    if kind == "yesno":
        return spec["gold"]
    if kind == "all_of":
        return ", ".join(spec["gold"])
    if kind == "json":
        return json.dumps(spec["gold"])
    raise ValueError(kind)


def to_jsonl(examples: list[Example]) -> str:
    return "".join(json.dumps(asdict(e), sort_keys=True) + "\n" for e in examples)
