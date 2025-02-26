# 2D_HTML Project Execution Manual

## 1. Download Node.js

Go to the official website to download Node.js: [Node.js Official Website](https://nodejs.org/en)

- **Recommended Version**:
  - Choose LTS (Long-Term Support) for stability
  - Or select the latest version (Current) for new features

- **Download the appropriate system version**:
  - Windows x64: Download the `.msi` installer
  - Other systems can choose the `.zip` archive

## 2. Verify Node.js Installation

Open the command prompt (cmd) or terminal and enter commands to check the installed Node.js and npm versions.

```sh
node -v  
npm -v    
```

If installed successfully, the corresponding version numbers should be displayed.

## 3. Navigate to the Project Directory

Change to the `2d_html` project directory.

```sh
cd ./2d_html/
```
## 4. Install Project Dependencies

Run the installation command in the project root directory to download all required dependencies.

```sh
npm install
```
## 5. Start the Web Server

Ensure that the `server.js` file is correctly configured, then run the startup command.

```sh
node server.js
```

If everything is set up correctly, the terminal should indicate that the web server has started successfully and provide an access URL (e.g., `http://localhost:3000`).

## 6. View the Project

Open a browser and enter the corresponding URL (e.g., `http://localhost:3000`) to view the project.

---

## Additional Operations

- If using `nodemon` to monitor changes, use the appropriate command to start the server.
- To stop the server, use the keyboard shortcut to terminate the process.

---

This is the basic execution manual for the 2D_HTML project. If you encounter any issues, please refer to the official documentation or contact the project maintainer!
