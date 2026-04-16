# Minimal MQTT Power History API

这是一个从原项目里“读取功率 MQTT topic”那段逻辑提取出来的最小版工具。

它做的事情很简单：

- 连原来的 MQTT broker
- 默认只订阅原来的功率 topic
- 把收到的消息记到本地 CSV
- 提供一个简单 HTTP API，按时间范围查数据或导出 CSV

不是 HA HTTP API，也不是全量 MQTT 抓包器。

## 配置文件

配置文件在：

```text
/mnt/hao_llm_experiment/Project1/Tool/HA_History_API/config.json
```

当前结构：

```json
{
  "mqtt": {
    "broker": "azure.nocolor.cc",
    "port": 1883,
    "username": "feiyue",
    "password": "123456789",
    "objects": [
      "homeassistant/sensor/chamberRoom_C1_right_light_current_power/state"
    ],
    "connect_timeout_seconds": 60
  },
  "query_defaults": {
    "timezone": "Australia/Sydney",
    "start_time": "2026-04-16 00:00:00",
    "end_time": "2026-04-16 23:59:59"
  }
}
```

你现在可以直接改两类东西：

- `mqtt.objects`
  订阅和查询允许使用的对象列表
- `query_defaults.timezone`
  默认时间解释时区
- `query_defaults.start_time`
  默认开始时间
- `query_defaults.end_time`
  默认结束时间

说明：

- 改 `mqtt.objects` 后，需要重启服务
- 改默认时间范围后，重启服务最稳妥
- 现在默认按悉尼时间处理，也就是 `Australia/Sydney`

## 默认配置说明

默认 broker 和账号直接沿用原项目：

```text
broker   = azure.nocolor.cc
port     = 1883
username = feiyue
password = 123456789
```

默认对象：

```text
homeassistant/sensor/chamberRoom_C1_right_light_current_power/state
```

## 启动

```bash
cd /mnt/hao_llm_experiment/Project1/Tool/HA_History_API
python app.py
```

默认监听：

```text
http://127.0.0.1:8010
```

也可以显式写成：

```bash
python app.py serve
```

## 命令行直接读取

如果你不想再开一个终端调 HTTP，现在可以直接用这些命令：

看当前配置：

```bash
python app.py config
```

看对象列表：

```bash
python app.py objects
```

看本地状态：

```bash
python app.py health
```

直接读历史摘要：

```bash
python app.py show --summary-only
```

直接读历史 JSON：

```bash
python app.py show
```

只看最近 20 条：

```bash
python app.py show --limit 20
```

直接导出 CSV：

```bash
python app.py export
```

导出到指定文件：

```bash
python app.py export --output ./today.csv
```

按时间范围导出：

```bash
python app.py export --start-time "2026-04-16 00:00:00" --end-time "2026-04-16 23:59:59"
```

## 本地文件

数据会保存到：

```text
logs/mqtt_messages_YYYYMMDD.csv
```

字段只有这些：

- `id`
- `timestamp`
- `topic`
- `payload`
- `qos`
- `retain`

## 接口

### 1. 看状态

```bash
curl http://127.0.0.1:8010/health
```

### 2. 看当前配置

```bash
curl http://127.0.0.1:8010/config
```

### 2. 看当前配置的对象

```bash
curl http://127.0.0.1:8010/objects
```

### 3. 查历史 JSON

如果不传 `objects`，就默认查当前配置的对象。

如果不传 `start_time/end_time`，就使用 `config.json` 里的默认开始时间和结束时间。

```bash
curl "http://127.0.0.1:8010/history"
```

如果配置了多个对象，也可以显式指定：

```bash
curl "http://127.0.0.1:8010/history?start_time=2026-04-16T00:00:00Z&end_time=2026-04-16T23:59:59Z&objects=homeassistant/sensor/chamberRoom_C1_right_light_current_power/state"
```

### 4. 导出 CSV

```bash
curl -L "http://127.0.0.1:8010/history.csv?start_time=2026-04-16T00:00:00Z&end_time=2026-04-16T23:59:59Z" -o power_history.csv
```

如果不传时间范围，就用配置里的默认开始时间和结束时间：

```bash
curl -L "http://127.0.0.1:8010/history.csv" -o power_history.csv
```

### 5. POST 查询 JSON

```bash
curl -X POST "http://127.0.0.1:8010/history/query" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2026-04-16T00:00:00Z",
    "end_time": "2026-04-16T23:59:59Z"
  }'
```

### 6. POST 导出 CSV

```bash
curl -X POST "http://127.0.0.1:8010/history/query.csv" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2026-04-16T00:00:00Z",
    "end_time": "2026-04-16T23:59:59Z",
    "filename": "power_history.csv"
  }' \
  -o power_history.csv
```

## 说明

- 现在只支持查询“配置里订阅的对象”
- 默认就是原项目的功率 topic
- 如果你想查多个对象，就改 `config.json` 里的 `mqtt.objects`
- `query_defaults.start_time` 和 `query_defaults.end_time` 现在默认按悉尼时间解释
