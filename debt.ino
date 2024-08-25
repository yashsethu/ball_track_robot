#include <BluetoothSerial.h>
#include <U8g2lib.h>

String device_name = "Stupid Chip";

U8G2_SSD1306_128X64_NONAME_F_SW_I2C
u8g2(U8G2_R0, 22, 21, U8X8_PIN_NONE);

BluetoothSerial BT;

bool shouldContinue = false;

void setup()
{
  Serial.begin(9600);
  u8g2.begin();

#ifndef CONFIG_BT_ENABLED
#ifndef CONFIG_BLUEDROID_ENABLED
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_5x7_mf);
  u8g2.drawStr(5, 13, "Bluetooth is not enabled!");
  u8g2.sendBuffer();
#endif
#endif

#ifndef CONFIG_BT_SPP_ENABLED
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_5x7_mf);
  u8g2.drawStr(5, 13, "Serial BT unavailable or disabled");
  u8g2.sendBuffer();
#else
  BT.begin(device_name); // Bluetooth device name
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_5x7_mf);
  String message = device_name + " started";
  u8g2.drawStr(5, 13, message.c_str());
  String message2 = "Waiting...";
  u8g2.drawStr(5, 23, message2.c_str());
  u8g2.sendBuffer();
  delay(1000);
  u8g2.clearBuffer();
  shouldContinue = true;
#endif
}

int y = 13; // Initial y-coordinate for the text
bool isConnected = false;
bool confirm = false;

void loop()
{
  if (shouldContinue == true)
  {
    if (BT.hasClient() && isConnected == false)
    {
      // Print the connection message on the OLED display at y = 13
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_5x7_mf);
      String message = "Device connected!";
      u8g2.drawStr(5, 13, message.c_str());
      String message2 = "Ready for recieving data";
      u8g2.drawStr(5, 23, message2.c_str());
      u8g2.sendBuffer();
      isConnected = true;
      confirm = true;
      y = 13;
    }

    if (BT.hasClient() == false && isConnected == true)
    {
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_5x7_mf);
      String message = "Device disconnected";
      u8g2.drawStr(5, 13, message.c_str());
      String message2 = "Waiting...";
      u8g2.drawStr(5, 23, message2.c_str());
      u8g2.sendBuffer();
      isConnected = false;
    }

    if (BT.available())
    {
      if (confirm == true)
      {
        u8g2.clearBuffer();
        confirm = false;
      }
      String message = "";
      while (BT.available())
      {
        char received = BT.read();
        Serial.write(received);
        message += String(received);
      }

      // Clear the buffer if the next message will exceed the display height
      if (y > 63)
      {
        u8g2.clearBuffer();
        y = 13; // Reset the y-coordinate
      }

      // Print the received data on the OLED display
      u8g2.setFont(u8g2_font_5x7_mf);
      u8g2.drawStr(5, y, message.c_str());
      u8g2.sendBuffer();
      y += 10; // Increment the y-coordinate for the next message
    }

    delay(20);
  }
}