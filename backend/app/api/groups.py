from fastapi import APIRouter, HTTPException
from app.models.group import GroupCreate, GroupUpdate, GroupData, GroupResponse
from app.models.common import ValidationResult
from app.store import store, new_id, save_store

router = APIRouter()


def _ok() -> ValidationResult:
    return ValidationResult(ok=True)


@router.post("/groups", response_model=GroupResponse, status_code=201)
def create_group(body: GroupCreate):
    gid = new_id()
    data = GroupData(id=gid, name=body.name, member_ids=body.member_ids)
    store.groups[gid] = data
    save_store()
    return GroupResponse(group=data, validation=_ok())


@router.patch("/groups/{gid}", response_model=GroupResponse)
def update_group(gid: str, body: GroupUpdate):
    if gid not in store.groups:
        raise HTTPException(404, "Group not found")
    data: GroupData = store.groups[gid]
    updates = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.member_ids is not None:
        updates["member_ids"] = body.member_ids
    data = data.model_copy(update=updates)
    store.groups[gid] = data
    save_store()
    return GroupResponse(group=data, validation=_ok())


@router.delete("/groups/{gid}", status_code=204)
def delete_group(gid: str):
    if gid not in store.groups:
        raise HTTPException(404, "Group not found")
    del store.groups[gid]
    save_store()
