#include <Arduino.h>
#include <Wire.h>
#include <U8g2lib.h>
#include <WiFi.h>

const int x = 5;
const int y = 13;

U8G2_SSD1306_128X64_NONAME_F_SW_I2C
u8g2(U8G2_R0, 22, 21, U8X8_PIN_NONE);

void print(int x, int y, char str[])
{
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_profont10_mr);
    u8g2.drawStr(x, y, str);
    u8g2.sendBuffer();
}

void setup()
{
    u8g2.begin();

    WiFi.mode(WIFI_STA); // Optional
    WiFi.disconnect();

    print(x, y, "Setup Done");
    delay(1000);
}

String findEncrypt(int ssid)
{
    if (WiFi.encryptionType(ssid) == WIFI_AUTH_OPEN)
    {
        return "Open";
    }
    else if (WiFi.encryptionType(ssid) == WIFI_AUTH_WEP)
    {
        return "WEP";
    }
    else if (WiFi.encryptionType(ssid) == WIFI_AUTH_WPA_PSK)
    {
        return "WPA";
    }
    else if (WiFi.encryptionType(ssid) == WIFI_AUTH_WPA2_PSK)
    {
        return "WPA2";
    }
    else if (WiFi.encryptionType(ssid) == WIFI_AUTH_WPA_WPA2_PSK)
    {
        return "WPA WPA2";
    }
    else if (WiFi.encryptionType(ssid) == WIFI_AUTH_WPA2_ENTERPRISE)
    {
        return "Enterprise";
    }
}

void loop()
{
    // Check if the button is pressed
    int n = WiFi.scanNetworks();
    if (n == 0)
    {
        u8g2.clearBuffer();
        u8g2.setFont(u8g2_font_profont10_mr);
        u8g2.drawStr(x, y, "No networks found");
    }
    else
    {
        u8g2.clearBuffer();
        u8g2.setFont(u8g2_font_profont10_mr);
        String str = String(n) + " networks found";
        u8g2.drawStr(x, y, str.c_str());
        for (int i = 0; i < n; ++i)
        {
            // Print SSID and RSSI for each network found
            String secured = findEncrypt(i);
            String str = String(i + 1) + ": " + WiFi.SSID(i) + " (" + WiFi.RSSI(i) + ") - " + secured;
            int y2 = y + 10 + (10 * i);
            u8g2.drawStr(x, y2, str.c_str());
            delay(10);
        }
    }
    u8g2.sendBuffer();
    // Wait a bit before checking the button state again
    delay(2000);
}