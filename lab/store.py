import json,uuid
from .auth import client
BUCKET="experiments"
def listing():
    return client().table("experiments").select("*").order("created_at",desc=True).limit(100).execute().data
def save(user,name,notes,settings,metrics,blob=None):
    eid=str(uuid.uuid4());obj=None
    if blob:
        if len(blob)>12*1024**2: raise ValueError("Result bundle exceeds 12 MiB.")
        obj=f"{user.id}/{eid}.npz"
        client().storage.from_(BUCKET).upload(obj,blob,{"content-type":"application/octet-stream"})
    try:
        client().table("experiments").insert(dict(id=eid,user_id=user.id,name=name[:100],
            notes=notes[:4000],settings=settings,metrics=metrics,object_path=obj)).execute()
    except Exception:
        if obj: client().storage.from_(BUCKET).remove([obj])
        raise
def download(record):
    return client().storage.from_(BUCKET).download(record["object_path"])
def delete(record):
    if record.get("object_path"): client().storage.from_(BUCKET).remove([record["object_path"]])
    client().table("experiments").delete().eq("id",record["id"]).execute()
def profile(user):
    r=client().table("profiles").select("*").eq("id",user.id).execute().data
    return r[0] if r else dict(display_name="",preferences={},bookmarks=[])
def update_profile(user,**fields):
    client().table("profiles").update(fields).eq("id",user.id).execute()
