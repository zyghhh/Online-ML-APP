import streamlit as st


class MultiPage:
    """Framework for combining multiple streamlit applications
    """

    def __init__(self) -> None:
        self.pages = []

    def add(self, title, func):
        self.pages.append(
            {
                'title': title,
                'function': func
            }
        )

    def run(self):
        page = st.sidebar.selectbox(
            'app navigation',
            self.pages,
            # Function to modify the display of the labels.
            format_func=lambda page: page['title']

        )
        page['function']()
