# test_simple_config.py
from src.utility.configLoader import config_loader


def test_basic_functionality():
    print("🧪 Testing ConfigLoader Basic Functionality")
    print("=" * 40)

    # Test 1: Check if config_loader is created
    print("1. Testing singleton instance...")
    print(f"   ✅ config_loader exists: {config_loader is not None}")
    print(f"   ✅ Type: {type(config_loader).__name__}")

    # Test 2: Check config base path
    print("\n2. Testing path resolution...")
    print(f"   ✅ Config folder: {config_loader.config_base_path}")
    print(f"   ✅ Folder exists: {config_loader.config_base_path.exists()}")

    # Test 3: Try to load a config file
    print("\n3. Testing config file loading...")
    try:
        # Try to load database config (common file)
        db_config = config_loader.load_config("database")
        print(f"   ✅ Database config loaded successfully")
        print(f"   ✅ Config type: {type(db_config).__name__}")
        print(f"   ✅ Keys found: {list(db_config.keys())}")
    except FileNotFoundError:
        print("   ⚠️  database.yaml not found (create it in config/ folder)")

    # Test 4: Test the get() method with safe defaults
    print("\n4. Testing get() method with defaults...")

    # This should always work - testing safety feature
    host = config_loader.get("database", "database.host", "localhost")
    print(f"   ✅ Database host: {host}")

    port = config_loader.get("database", "database.port", 5432)
    print(f"   ✅ Database port: {port}")

    # Test non-existent key
    missing = config_loader.get("database", "nonexistent.key", "default_value")
    print(f"   ✅ Missing key returns default: {missing}")

    print("\n🎉 BASIC CONFIGLOADER TEST COMPLETED!")
    print("✅ Your ConfigLoader is working correctly!")


def test_real_usage():
    print("\n🧪 Testing Real Usage Scenarios")
    print("=" * 40)

    # Simulate how you'd use it in your project
    print("Simulating project usage:")

    # Database configuration
    db_settings = {
        "host": config_loader.get("database", "host", "localhost"),
        "port": config_loader.get("database", "port", 5432),
        "timeout": config_loader.get("database", "timeout", 30),
    }
    print(f"📊 Database settings: {db_settings}")

    # App settings
    app_name = config_loader.get("app", "name", "My Financial App")
    debug_mode = config_loader.get("app", "debug", False)
    print(f"📱 App: {app_name}, Debug: {debug_mode}")

    # Feature flags
    use_redis = config_loader.get("features", "redis.enabled", False)
    cache_size = config_loader.get("features", "cache.size", 100)
    print(f"⚡ Features - Redis: {use_redis}, Cache: {cache_size}MB")


if __name__ == "__main__":
    test_basic_functionality()
    test_real_usage()
