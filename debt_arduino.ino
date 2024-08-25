#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <U8g2lib.h>

const char *ssid = "Free Public Access Almaden-CC";
const char *password = "dragonwarrior";
String lastFetchedDebt = "Fetching...";

U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2(U8G2_R0, 22, 21, U8X8_PIN_NONE);

void drawCenteredString(String text, int y)
{
    int x = (u8g2.getDisplayWidth() - u8g2.getStrWidth(text.c_str())) / 2;
    u8g2.drawStr(x, y, text.c_str());
}

String formatWithCommas(double value)
{
    char buffer[50];
    dtostrf(value, 1, 2, buffer);

    String str = String(buffer);
    int len = str.length();
    int decPos = str.indexOf('.');
    if (decPos == -1)
        decPos = len;

    for (int i = decPos - 3; i > 0; i -= 3)
    {
        str = str.substring(0, i) + ',' + str.substring(i);
    }

    return str;
}

double findMostRecentValue(const char *data)
{
    StaticJsonDocument<1024> doc;
    deserializeJson(doc, data);

    double mostRecentValue = 0;
    String mostRecentDate;

    for (JsonVariant value : doc["data"].as<JsonArray>())
    {
        String recordDate = value["record_date"];
        double debtValue = atof(value["tot_pub_debt_out_amt"]);

        if (recordDate > mostRecentDate)
        {
            mostRecentDate = recordDate;
            mostRecentValue = debtValue;
        }
    }

    return mostRecentValue;
}

String getDebt()
{
    if (WiFi.status() != WL_CONNECTED)
    {
        WiFi.begin(ssid);
        return lastFetchedDebt;
    }

    HTTPClient http;
    http.begin("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny?fields=record_date,%20tot_pub_debt_out_amt&sort=-record_date");
    int httpCode = http.GET();

    if (httpCode > 0)
    {
        String payload = http.getString();
        double mostRecentValue = findMostRecentValue(payload.c_str());
        lastFetchedDebt = formatWithCommas(mostRecentValue);
    }

    http.end();
    return lastFetchedDebt;
}

void setup()
{
    Serial.begin(115200);
    u8g2.begin();
    delay(1000);

    WiFi.begin(ssid, password);
    String loading = ".";

    while (WiFi.status() != WL_CONNECTED)
    {
        u8g2.clearBuffer();
        u8g2.setFont(u8g2_font_5x7_mf);
        String message = "Connecting" + loading;
        drawCenteredString(message, 13);
        u8g2.sendBuffer();
        if (loading == "...")
        {
            loading = ".";
        }
        loading += ".";
        delay(500);
    }

    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_5x7_mf);
    String message = "Connected!";
    drawCenteredString(message, 13);
    u8g2.sendBuffer();
    delay(1000);
}

void loop()
{
    String debt = getDebt();
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_5x7_mf);

    String message = "US NATIONAL DEBT:";
    drawCenteredString(message, 13);

    String debtValue = "$ " + debt;
    Serial.print(debtValue);
    drawCenteredString(debtValue, 33);

    u8g2.sendBuffer();
    delay(1000);
}