"""Run ONLY against a disposable test project, after migrations, with two confirmed test users.
Environment: SUPABASE_URL, SUPABASE_KEY, TEST_EMAIL_A/B, TEST_PASSWORD_A/B.
Creates one record, verifies cross-user access is blocked, then cleans up.
"""
import os,uuid
from supabase import create_client
def login(suffix):
    c=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_KEY"])
    r=c.auth.sign_in_with_password({"email":os.environ["TEST_EMAIL_"+suffix],"password":os.environ["TEST_PASSWORD_"+suffix]})
    return c,r.user
a,ua=login("A");b,ub=login("B")
assert ua.id!=ub.id
eid=str(uuid.uuid4());obj=f"{ua.id}/{eid}.npz";inserted=False
try:
    a.table("experiments").insert(dict(id=eid,user_id=ua.id,name="RLS integration test")).execute();inserted=True
    assert a.table("experiments").select("id").eq("id",eid).execute().data
    assert not b.table("experiments").select("id").eq("id",eid).execute().data
    assert not b.table("experiments").update({"name":"intrusion"}).eq("id",eid).execute().data
    assert not b.table("profiles").select("*").eq("id",ua.id).execute().data
    blocked=False
    try:b.table("experiments").insert(dict(user_id=ua.id,name="spoofed")).execute()
    except Exception:blocked=True
    assert blocked,"Ownership spoofing succeeded"
    a.storage.from_("experiments").upload(obj,b"test",{"content-type":"application/octet-stream"})
    blocked=False
    try:b.storage.from_("experiments").download(obj)
    except Exception:blocked=True
    assert blocked,"Cross-user object read succeeded"
    print("Owner and cross-user RLS checks passed.")
finally:
    try:a.storage.from_("experiments").remove([obj])
    except Exception:pass
    if inserted:a.table("experiments").delete().eq("id",eid).execute()
    a.auth.sign_out();b.auth.sign_out()
