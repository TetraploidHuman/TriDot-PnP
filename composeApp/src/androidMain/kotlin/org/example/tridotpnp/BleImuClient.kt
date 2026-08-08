package org.example.tridotpnp

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/**
 * BLE client for wearable nodes.
 *
 * Packet (little-endian, 28 bytes):
 *  magic0 magic1 nodeId seq
 *  ax ay az (m/s^2) gx gy gz (deg/s)  float32 each
 *  timestampMs uint32
 *
 * Service/characteristic UUIDs are project-local defaults; firmware should match.
 */
class BleImuClient(private val context: Context) {
    companion object {
        private const val TAG = "BleImuClient"
        val SERVICE_UUID: UUID = UUID.fromString("6e400001-b5a3-f393-e0a9-e50e24dcca9e")
        val IMU_CHAR_UUID: UUID = UUID.fromString("6e400003-b5a3-f393-e0a9-e50e24dcca9e")
        val CCCD_UUID: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
        private const val MAGIC0: Byte = 0x54 // 'T'
        private const val MAGIC1: Byte = 0x44 // 'D'
        const val PACKET_SIZE = 28
    }

    private val adapter: BluetoothAdapter? =
        (context.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager)?.adapter

    private val _samples = MutableStateFlow<ImuSample?>(null)
    val samples: StateFlow<ImuSample?> = _samples.asStateFlow()

    private val _scanning = MutableStateFlow(false)
    val scanning: StateFlow<Boolean> = _scanning.asStateFlow()

    private val _connectedDevices = MutableStateFlow<List<String>>(emptyList())
    val connectedDevices: StateFlow<List<String>> = _connectedDevices.asStateFlow()

    private val gatts = ConcurrentHashMap<String, BluetoothGatt>()
    private var scanCallback: ScanCallback? = null

    @SuppressLint("MissingPermission")
    fun startScan(namePrefix: String = "TriDot") {
        val scanner = adapter?.bluetoothLeScanner ?: return
        stopScan()
        scanCallback = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                val name = result.device.name ?: return
                if (!name.startsWith(namePrefix)) return
                connect(result.device)
            }
        }
        _scanning.value = true
        try {
            scanner.startScan(scanCallback)
        } catch (t: Throwable) {
            Log.w(TAG, "startScan failed: ${t.message}")
            _scanning.value = false
        }
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        val scanner = adapter?.bluetoothLeScanner
        scanCallback?.let {
            try {
                scanner?.stopScan(it)
            } catch (_: Throwable) {
            }
        }
        scanCallback = null
        _scanning.value = false
    }

    @SuppressLint("MissingPermission")
    fun connect(device: BluetoothDevice) {
        val addr = device.address
        if (gatts.containsKey(addr)) return
        val gatt = device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
        gatts[addr] = gatt
    }

    @SuppressLint("MissingPermission")
    fun disconnectAll() {
        stopScan()
        gatts.values.forEach {
            try {
                it.disconnect()
                it.close()
            } catch (_: Throwable) {
            }
        }
        gatts.clear()
        _connectedDevices.value = emptyList()
    }

    fun parsePacket(bytes: ByteArray, fallbackNodeId: Int = 0): ImuSample? {
        if (bytes.size < PACKET_SIZE) return null
        if (bytes[0] != MAGIC0 || bytes[1] != MAGIC1) return null
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        buf.position(2)
        val nodeId = buf.get().toInt() and 0xFF
        buf.get() // seq
        val ax = buf.float
        val ay = buf.float
        val az = buf.float
        val gx = buf.float
        val gy = buf.float
        val gz = buf.float
        val ts = if (bytes.size >= 32) {
            buf.int.toLong() and 0xFFFFFFFFL
        } else {
            0L
        }
        return ImuSample(
            nodeId = if (nodeId != 0) nodeId else fallbackNodeId,
            ax = ax,
            ay = ay,
            az = az,
            gx = gx,
            gy = gy,
            gz = gz,
            timestampMs = if (ts != 0L) ts else System.currentTimeMillis()
        )
    }

    private val gattCallback = object : BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                gatt.discoverServices()
                publishConnected()
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                gatts.remove(gatt.device.address)
                try {
                    gatt.close()
                } catch (_: Throwable) {
                }
                publishConnected()
            }
        }

        @SuppressLint("MissingPermission")
        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            val service = gatt.getService(SERVICE_UUID) ?: return
            val characteristic = service.getCharacteristic(IMU_CHAR_UUID) ?: return
            gatt.setCharacteristicNotification(characteristic, true)
            val descriptor = characteristic.getDescriptor(CCCD_UUID)
            if (descriptor != null) {
                if (Build.VERSION.SDK_INT >= 33) {
                    gatt.writeDescriptor(descriptor, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
                } else {
                    @Suppress("DEPRECATION")
                    descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                    @Suppress("DEPRECATION")
                    gatt.writeDescriptor(descriptor)
                }
            }
        }

        @Deprecated("Deprecated in Java")
        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            @Suppress("DEPRECATION")
            val value = characteristic.value ?: return
            handleBytes(value)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            handleBytes(value)
        }
    }

    private fun handleBytes(value: ByteArray) {
        val sample = parsePacket(value) ?: return
        _samples.value = sample
    }

    @SuppressLint("MissingPermission")
    private fun publishConnected() {
        _connectedDevices.value = gatts.keys.toList()
    }
}
