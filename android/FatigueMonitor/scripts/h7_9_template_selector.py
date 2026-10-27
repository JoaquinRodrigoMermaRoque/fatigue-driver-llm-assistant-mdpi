#!/usr/bin/env python3
import argparse
import json
import sys

TEMPLATES = [
    "repeated_primary_with_recent_events",
    "recent_events_changed_from_history",
    "recovery_after_critical",
]
PRIMARY = ["eye_closure", "yawns", "recovery"]
SECONDARY = ["none", "yawns_and_nod", "gaze_away"]

APPROVED = {
    (
        "repeated_primary_with_recent_events",
        "eye_closure",
        "yawns_and_nod",
    ): (
        "Joaquín, el cierre ocular prolongado volvió a presentarse; "
        "además, se registraron bostezos y un cabeceo recientemente."
    ),
    (
        "recent_events_changed_from_history",
        "yawns",
        "gaze_away",
    ): (
        "Joaquín, los bostezos se han repetido y también apareció una "
        "desviación prolongada de la mirada, un cambio respecto a las "
        "señales leves previas."
    ),
    (
        "recovery_after_critical",
        "recovery",
        "none",
    ): (
        "Joaquín, las señales se mantienen en recuperación después de un "
        "episodio crítico previo, sin nuevos bostezos ni cabeceos."
    ),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "template": {"type": "string", "enum": TEMPLATES},
        "primary": {"type": "string", "enum": PRIMARY},
        "secondary": {"type": "string", "enum": SECONDARY},
    },
    "required": ["template", "primary", "secondary"],
    "additionalProperties": False,
}

STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "variant": {"type": "string", "enum": ["a", "b", "c"]},
    },
    "required": ["variant"],
    "additionalProperties": False,
}

STYLE_RENDERERS = {
    "repeated_primary_with_recent_events": {
        "a": (
            "Joaquín, el cierre ocular prolongado volvió a presentarse; "
            "además, se registraron bostezos y un cabeceo recientemente."
        ),
        "b": (
            "Joaquín, volvió a registrarse un cierre ocular prolongado, "
            "junto con bostezos y un cabeceo en los eventos recientes."
        ),
        "c": (
            "Joaquín, se repitió el cierre ocular prolongado y también se "
            "registraron bostezos y un cabeceo recientemente."
        ),
    },
    "recent_events_changed_from_history": {
        "a": (
            "Joaquín, los bostezos se han repetido y también apareció una "
            "desviación prolongada de la mirada, un cambio respecto a las "
            "señales leves previas."
        ),
        "b": (
            "Joaquín, ahora se registran bostezos repetidos y una desviación "
            "prolongada de la mirada, a diferencia de las señales leves previas."
        ),
        "c": (
            "Joaquín, los eventos recientes incluyen bostezos repetidos y una "
            "desviación prolongada de la mirada, distintos al historial leve previo."
        ),
    },
    "recovery_after_critical": {
        "a": (
            "Joaquín, las señales se mantienen en recuperación después de un "
            "episodio crítico previo, sin nuevos bostezos ni cabeceos."
        ),
        "b": (
            "Joaquín, se observa una recuperación sostenida tras el episodio "
            "crítico previo, sin nuevos bostezos ni cabeceos."
        ),
        "c": (
            "Joaquín, la recuperación continúa después del episodio crítico "
            "anterior y no se han registrado nuevos bostezos ni cabeceos."
        ),
    },
}


def render(selection):
    key = (
        selection.get("template"),
        selection.get("primary"),
        selection.get("secondary"),
    )
    return APPROVED.get(key)


def parse_completion_response(path):
    with open(path, encoding="utf-8") as f:
        response = json.load(f)

    raw = response.get("content", "")
    if not isinstance(raw, str):
        raise ValueError("completion content is not a string")

    selection = json.loads(raw)
    if not isinstance(selection, dict):
        raise ValueError("selection is not a JSON object")

    return selection


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("schema")
    sub.add_parser("style-schema")

    render_parser = sub.add_parser("render-response")
    render_parser.add_argument("response_json")

    style_render_parser = sub.add_parser("render-style-response")
    style_render_parser.add_argument("template")
    style_render_parser.add_argument("response_json")

    args = parser.parse_args()

    if args.command == "schema":
        print(json.dumps(SCHEMA, ensure_ascii=False, separators=(",", ":")))
        return

    if args.command == "style-schema":
        print(json.dumps(STYLE_SCHEMA, ensure_ascii=False, separators=(",", ":")))
        return

    if args.command == "render-style-response":
        try:
            with open(args.response_json, encoding="utf-8") as f:
                response = json.load(f)
            raw = response.get("content", "")
            selection = json.loads(raw)
            variant = selection.get("variant")
            rendered = STYLE_RENDERERS.get(args.template, {}).get(variant)
        except Exception as exc:
            print(json.dumps({
                "accepted": False,
                "reason": f"parse_error:{type(exc).__name__}",
                "template": args.template,
                "variant": None,
                "rendered": None,
            }, ensure_ascii=False))
            return

        if rendered is None:
            print(json.dumps({
                "accepted": False,
                "reason": "unsupported_variant",
                "template": args.template,
                "variant": variant,
                "rendered": None,
            }, ensure_ascii=False))
            return

        print(json.dumps({
            "accepted": True,
            "reason": "approved_style_variant",
            "template": args.template,
            "variant": variant,
            "rendered": rendered,
        }, ensure_ascii=False))
        return

    try:
        selection = parse_completion_response(args.response_json)
        rendered = render(selection)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "reason": f"parse_error:{type(exc).__name__}",
                    "selection": None,
                    "rendered": None,
                },
                ensure_ascii=False,
            )
        )
        return

    if rendered is None:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "reason": "unsupported_combination",
                    "selection": selection,
                    "rendered": None,
                },
                ensure_ascii=False,
            )
        )
        return

    print(
        json.dumps(
            {
                "accepted": True,
                "reason": "approved_template",
                "selection": selection,
                "rendered": rendered,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
