# TriDot-PnP

可穿戴视觉–惯性位姿感知：Android 相机伴侣应用 + ESP32-C3 节点固件。

## 仓库结构

| 路径 | 说明 |
|------|------|
| [`composeApp/`](./composeApp) | Android 伴侣应用（Compose）：RGB LED PnP、多节点跟踪、ESKF、BLE |
| [`firmware/`](./firmware) | **ESP-IDF** 固件（**ESP32-C3**：双 IMU、LED 身份编码、BLE） |
| [`firmware/micropython/`](./firmware/micropython) | 可选 MicroPython LED 点亮脚本 |

## Android 应用

```shell
.\gradlew.bat :composeApp:assembleDebug
```

每次推送到 `main` 时，CI 会构建 debug APK。打标签发布（`v*`）时会产出可安装的 release APK。

> 注意：若未配置正式签名密钥，release 包会回退为 debug 签名，适合内测，不适合上架商店。

## ESP32-C3 固件

详见 [`firmware/README.md`](./firmware/README.md)。

```shell
cd firmware
idf.py set-target esp32c3
idf.py build flash monitor
```

## 协议

共享 BLE + LED 身份格式：[`firmware/PROTOCOL.md`](./firmware/PROTOCOL.md)。
