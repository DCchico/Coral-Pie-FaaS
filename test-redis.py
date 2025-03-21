import redis
import time

# Connect to Redis
redis_client = redis.Redis(
    host="128.110.217.193",
    port=6379,
    decode_responses=True
)

# Enable keyspace notifications
redis_client.config_set('notify-keyspace-events', 'KA')

# Create a pubsub object
pubsub = redis_client.pubsub()

# Key to monitor
key = "incamera:test:1"
pubsub.subscribe(f"__keyspace@0__:{key}")
pubsub.subscribe("__keyevent@0__:set")

print(f"Waiting for activity on key {key}")

# In another terminal, run: redis-cli set incamera:test:1 "hello"
while True:
    message = pubsub.get_message(timeout=1.0)
    if message:
        print(f"Received message: {message}")
    time.sleep(0.1)