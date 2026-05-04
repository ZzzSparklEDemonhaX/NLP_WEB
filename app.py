from pathlib import Path

import streamlit as st

from modules.registry import APP_REGISTRY
from modules.ui import app_card_grid, hero_section, metric_strip, render_footer, render_sidebar


ROOT_DIR = Path(__file__).parent
STYLE_PATH = ROOT_DIR / "assets" / "styles.css"


def load_css() -> None:
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="NLP Lab Studio",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()

    selected_app = render_sidebar(APP_REGISTRY)

    if selected_app is None:
        hero_section()
        metric_strip()
        app_card_grid(APP_REGISTRY)
        render_footer()
        return

    selected_app.render()


if __name__ == "__main__":
    main()
