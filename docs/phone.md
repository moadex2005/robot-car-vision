# Use a phone

[Back to the README](../README.md)

A phone can do two different jobs: be the camera, or show the page.

## The phone as the camera

1. Install the **IP Webcam** app on the Android phone.
2. Put the phone and the laptop on the same network. The simplest way is to turn on the phone's own hotspot and join it from the laptop.
3. In IP Webcam, scroll down and tap **Start server**. The app shows an address such as `http://192.168.43.1:8080`. You do not need to set the video resolution: the program asks the phone for a small picture itself (`stream.phone_picture: auto` in `config.yaml`).
4. Start the demo page and type that address into the **Camera** box. (Or put the address plus `/video` into `stream.phone_url` in `config.yaml`; an address typed on the page is remembered in `data/camera.json` and wins at the next start.)
5. Test it:

   ```powershell
   .venv\Scripts\python.exe -m car_vision.stream --source phone
   ```

   Press `c` to show a millisecond clock. Point the phone at the clock: the clock shows now, the video of the clock shows then, and the difference is the delay. Press `q` to quit.

## Open the page on your phone

The laptop's own page (`http://127.0.0.1:5000/`) opens only on the laptop, with no password, and that never changes. A phone gets a second door of its own, which is off until you switch it on:

1. Add a line `PAGE_PASSWORD=...` to `.env`: a password of your own, 8 characters or more. It stays on this laptop.
2. In `config.yaml` set `phone_door.enabled: true`.
3. Start the program. It prints a line like `Phone door: https://192.168.43.105:5443/`. The first time, Windows Firewall asks whether Python may be reached from the network: allow it for private networks.
4. Put the phone on the same network as the laptop (the phone's own hotspot works), then open that address on the phone. The **Camera** box on the laptop's page shows the address and a QR code to scan.
5. The phone warns that it does not know the certificate. That is expected: the laptop made the certificate itself (in `data/door/`, with the OpenSSL that comes with Git). Choose to continue, then type the password.

What to know:

- The door speaks HTTPS, so the password and the video cannot be read by others on the network, and the phone's browser allows the microphone.
- A phone stays signed in for 12 hours.
- After 5 wrong passwords from one phone it must wait 5 minutes, and after 30 from all phones together everyone must.
- When something is missing (no password, no OpenSSL, no network) the program says so in one line and runs on without the door.
- `app.host` must stay `127.0.0.1`: the program refuses to start with any other value, because that would open the page to the network without a password.
