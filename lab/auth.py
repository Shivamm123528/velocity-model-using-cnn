"""Each browser session owns its client; never globally cache authenticated clients."""
import streamlit as st
from supabase import create_client, ClientOptions
def configured():
    try: return bool(st.secrets["supabase"]["url"] and st.secrets["supabase"]["key"])
    except (KeyError,FileNotFoundError): return False
def client():
    if not configured(): raise RuntimeError("Supabase secrets are not configured.")
    if "_sb" not in st.session_state:
        st.session_state["_sb"]=create_client(st.secrets["supabase"]["url"],
            st.secrets["supabase"]["key"],options=ClientOptions(auto_refresh_token=False,persist_session=True))
    return st.session_state["_sb"]
def identity():
    if not st.session_state.get("_tokens"): return None
    try:
        c=client(); a,r=st.session_state["_tokens"]
        response=c.auth.set_session(a,r)  # refreshes expired access tokens
        if not response.session: raise RuntimeError("Session expired.")
        st.session_state["_tokens"]=(response.session.access_token,response.session.refresh_token)
        return c.auth.get_user().user
    except Exception:
        clear(); st.warning("Your session ended. Please sign in again.");return None
def remember(response):
    if response.session:
        st.session_state["_tokens"]=(response.session.access_token,response.session.refresh_token)
def clear():
    # Clear private arrays, loaded experiments and profile widgets on logout/account change.
    for k in list(st.session_state):
        if k not in ("nav",): del st.session_state[k]
def panel():
    st.subheader("Your research workspace")
    st.caption("Guest tools work without an account. Sign in to save experiments. Login survives reruns, not a browser reload.")
    if not configured():
        st.info("Guest demo is ready. Add Supabase secrets to enable registration and saved work.");return
    login,register,verify,reset=st.tabs(["Login","Register","Verify email","Reset password"])
    with login:
        with st.form("login",clear_on_submit=True):
            email=st.text_input("Email");pw=st.text_input("Password",type="password")
            submit=st.form_submit_button("Sign in")
        if submit:
            try:
                response=client().auth.sign_in_with_password({"email":email.strip(),"password":pw})
                clear();remember(response);st.rerun()
            except Exception: st.error("Sign-in failed. Check your details and email verification.")
    with register:
        with st.form("register",clear_on_submit=True):
            name=st.text_input("Display name",max_chars=80)
            em=st.text_input("Email address");p=st.text_input("Create password",type="password")
            p2=st.text_input("Confirm password",type="password")
            ok=st.form_submit_button("Create account")
        if ok:
            if not name.strip() or "@" not in em or len(p)<10 or p!=p2:
                st.error("Enter a name, email and matching passwords of at least 10 characters.")
            else:
                try:
                    r=client().auth.sign_up({"email":em.strip(),"password":p,"options":{"data":{"display_name":name.strip()}}})
                    if r.session: clear();remember(r);st.rerun()
                    st.success("Check your email for the verification code. Use the Verify email tab.")
                except Exception: st.error("Registration could not complete. Check project auth settings or try again later.")
    with verify:
        with st.form("verify",clear_on_submit=True):
            em=st.text_input("Registered email");token=st.text_input("Email confirmation code")
            ok=st.form_submit_button("Verify email")
        if ok:
            try:
                r=client().auth.verify_otp({"email":em.strip(),"token":token.strip(),"type":"email"})
                clear();remember(r);st.rerun()
            except Exception: st.error("Code invalid or expired. Check the confirmation email.")
    with reset:
        st.caption("Request an email, then enter its recovery code and a new password here.")
        with st.form("request_reset",clear_on_submit=True):
            em=st.text_input("Recovery email");ok=st.form_submit_button("Send recovery email")
        if ok:
            try: client().auth.reset_password_for_email(em.strip());st.success("If the account is eligible, a recovery email will arrive.")
            except Exception: st.error("Recovery request failed. Try again later.")
        with st.form("complete_reset",clear_on_submit=True):
            em=st.text_input("Account email");code=st.text_input("Recovery code")
            pw=st.text_input("New password",type="password");pw2=st.text_input("Repeat new password",type="password")
            ok=st.form_submit_button("Set new password")
        if ok:
            if len(pw)<10 or pw!=pw2: st.error("Use matching passwords with at least 10 characters.")
            else:
                try:
                    r=client().auth.verify_otp({"email":em.strip(),"token":code.strip(),"type":"recovery"})
                    client().auth.update_user({"password":pw});client().auth.sign_out()
                    clear();st.success("Password updated. Sign in with your new password.")
                except Exception: st.error("Recovery failed. Request a new code.")
