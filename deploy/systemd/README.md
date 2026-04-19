Recommended `systemd` units for this project.

Why these units run the Python scripts directly:
- `edge.sh start` and `riotee_system_manager.py start` both launch child/background processes.
- `systemd` is more reliable when it manages the real long-running process as `MAINPID`.
- This gives you boot auto-start plus automatic restart after crashes.

Install:

```bash
sudo cp /home/hao/Desktop/Project1/deploy/systemd/greenhouse-edge.service /etc/systemd/system/
sudo cp /home/hao/Desktop/Project1/deploy/systemd/riotee-collector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable greenhouse-edge.service
sudo systemctl enable riotee-collector.service
sudo systemctl start greenhouse-edge.service
sudo systemctl start riotee-collector.service
```

Useful commands:

```bash
sudo systemctl status greenhouse-edge.service
sudo systemctl status riotee-collector.service
sudo journalctl -u greenhouse-edge.service -f
sudo journalctl -u riotee-collector.service -f
sudo systemctl restart greenhouse-edge.service
sudo systemctl restart riotee-collector.service
```
