import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def app():
    local_css("style/style.css")
    # ---- LOAD ASSETS ----
    lottie_coding = load_lottieurl(
        "https://assets4.lottiefiles.com/packages/lf20_4kmUDEKo63.json"
    )
    lottie_coding2 = load_lottieurl(
        "https://assets10.lottiefiles.com/packages/lf20_fd83HLtqZt.json"
    )
    img_sphere = Image.open("images/sphere.jpg")
    img_phase_separation = Image.open("images/phase_separation.jpg")
    img_nano = Image.open("images/nano.jpg")

    # ---- HEADER SECTION ----
    with st.container():
        left_column, right_column = st.columns([3, 1])
        with left_column:
            st.subheader("Hi, this is an Online-Machine-Learning platform !")
            st.write(
                "A tool which can help you study online machine learning and deal data stream with concept drift"
            )

            st.write("[Learn More >](https://github.com/online-ml)")
        with right_column:
            st_lottie(lottie_coding, height=300, key="coding")
    # ---- WHAT I DO ----
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("What I do")
            st.write("##")
            st.write(
                """

                更新中


                """
            )

        with right_column:
            st_lottie(lottie_coding2, height=300, key="coding2")

    # ---- CONTACT ----
    with st.container():
        st.write("---")
        st.header("Get In Touch With Me!")
        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
            st.empty()
