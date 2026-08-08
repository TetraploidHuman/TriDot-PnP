# TriDot-PnP

Wearable visual–inertial pose sensing: Android camera app + ESP32-C3 node firmware.

## Repository layout

| Path | Description |
|------|-------------|
| [`composeApp/`](./composeApp) | Android companion app (Compose): RGB LED PnP, multi-node tracking, ESKF, BLE |
| [`firmware/`](./firmware) | **ESP-IDF** firmware for **ESP32-C3** (dual IMU, LED ID encode, BLE) |
| [`firmware/micropython/`](./firmware/micropython) | Optional MicroPython LED bring-up script |

## Android app

```shell
.\gradlew.bat :composeApp:assembleDebug
```

CI builds debug APK on every `main` push. Tagged releases (`v*`) publish installable APKs.

## ESP32-C3 firmware

See [`firmware/README.md`](./firmware/README.md).

```shell
cd firmware
idf.py set-target esp32c3
idf.py build flash monitor
```

## Protocol

Shared BLE + LED identity format: [`firmware/PROTOCOL.md`](./firmware/PROTOCOL.md).
