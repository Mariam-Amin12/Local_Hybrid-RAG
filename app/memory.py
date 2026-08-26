import json
import uuid

import redis

from app.schemas.chat import ChatMessage


class RedisSessionStore:

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def _key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def new_session_id(self) -> str:
        session_id = uuid.uuid4().hex

        print(
            f"[memory] Created session {session_id}",
            flush=True,
        )

        return session_id

    def get_history(self, session_id: str) -> list[ChatMessage]:
        key = self._key(session_id)

        messages = self.redis.lrange(key, 0, -1)

        history = [
            ChatMessage(**json.loads(message))
            for message in messages
        ]

        print(
            f"[memory] Loaded {len(history)} messages "
            f"for session {session_id}",
            flush=True,
        )

        return history

    def save_turn(
        self,
        session_id: str,
        user_msg: ChatMessage,
        ai_msg: ChatMessage,
    ) -> None:

        key = self._key(session_id)

        self.redis.rpush(
            key,
            json.dumps(user_msg.model_dump()),
            json.dumps(ai_msg.model_dump()),
        )

        print(
            f"[memory] Saved turn for session {session_id}; "
            f"messages={self.redis.llen(key)}",
            flush=True,
        )

    def clear_session(self, session_id: str) -> None:
        key = self._key(session_id)

        self.redis.delete(key)

        print(
            f"[memory] Cleared session {session_id}",
            flush=True,
        )