#!/usr/bin/env python3
"""
Environment setup script for NEXA AI IoT Project
This script helps users set up their environment by copying example files
and providing instructions for setting up API keys and credentials.
"""

import os
import shutil

def setup_environment():
    print("🔧 Setting up NEXA AI IoT Project Environment")
    print("=" * 50)
    
    # Copy .env.example to .env if it doesn't exist
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file from .env.example")
            print("   Please edit .env with your actual API keys")
        else:
            print("⚠️  .env.example not found. Creating a new .env file")
            with open('.env', 'w') as f:
                f.write("# API Keys - Keep these secret!\n")
                f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
                f.write("BLYNK_AUTH_TOKEN=your_blynk_auth_token_here\n")
            print("✅ Created .env file")
            print("   Please edit .env with your actual API keys")
    else:
        print("✅ .env file already exists")
    
    # Copy credentials.h.example to credentials.h if it doesn't exist
    if not os.path.exists('credentials.h'):
        if os.path.exists('credentials.h.example'):
            shutil.copy('credentials.h.example', 'credentials.h')
            print("✅ Created credentials.h file from credentials.h.example")
            print("   Please edit credentials.h with your actual WiFi credentials")
        else:
            print("⚠️  credentials.h.example not found. Creating a new credentials.h file")
            with open('credentials.h', 'w') as f:
                f.write("#ifndef CREDENTIALS_H\n")
                f.write("#define CREDENTIALS_H\n\n")
                f.write("// WiFi credentials\n")
                f.write('#define WIFI_SSID "your_wifi_ssid_here"\n')
                f.write('#define WIFI_PASS "your_wifi_password_here"\n\n')
                f.write("#endif\n")
            print("✅ Created credentials.h file")
            print("   Please edit credentials.h with your actual WiFi credentials")
    else:
        print("✅ credentials.h file already exists")
    
    print("\n" + "=" * 50)
    print("📋 Next steps:")
    print("1. Edit .env file with your actual API keys")
    print("2. Edit credentials.h file with your actual WiFi credentials")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run the server: python server_v4.py")

if __name__ == "__main__":
    setup_environment()