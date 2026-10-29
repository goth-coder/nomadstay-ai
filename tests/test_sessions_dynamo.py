from collections import deque

from api.sessions import DynamoSessionStore


class FakeTable:
    """The two DynamoDB calls the store uses, over a dict."""

    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        self.items[Item["session_id"]] = Item

    def get_item(self, Key):
        item = self.items.get(Key["session_id"])
        return {"Item": item} if item else {}


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_saved_context_comes_back_in_another_store_on_the_same_table():
    table = FakeTable()
    session_id, context = DynamoSessionStore(table).create()
    context.append({"role": "user", "content": "Porto, wifi"})
    DynamoSessionStore(table).save(session_id, context)

    other_task = DynamoSessionStore(table)  # another task: nothing in memory, same table
    assert other_task.get(session_id) == deque([{"role": "user", "content": "Porto, wifi"}])


def test_a_session_is_stored_only_when_saved():
    store = DynamoSessionStore(FakeTable())
    session_id, _ = store.create()
    assert store.get(session_id) is None


def test_expired_session_is_gone_even_before_dynamodb_deletes_it():
    clock = Clock()
    store = DynamoSessionStore(FakeTable(), ttl_seconds=60, clock=clock)
    session_id, context = store.create()
    store.save(session_id, context)

    clock.now += 61
    assert store.get(session_id) is None


def test_every_save_renews_the_ttl():
    clock = Clock()
    store = DynamoSessionStore(FakeTable(), ttl_seconds=60, clock=clock)
    session_id, context = store.create()
    store.save(session_id, context)
    clock.now += 50
    store.save(session_id, context)
    clock.now += 50
    assert store.get(session_id) is not None
    