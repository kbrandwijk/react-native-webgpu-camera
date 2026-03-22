import Reactotron from "reactotron-react-native";

Reactotron.configure({
  onCommand: (cmd) => {
    console.log("[Reactotron] incoming command:", cmd.type, JSON.stringify(cmd.payload));
  },
}) // controls connection & communication settings
  .useReactNative() // add all built-in react native plugins
  .connect(); // let's connect!

Reactotron.onCustomCommand({
  command: "ping",
  handler: () => {
    console.log("[ping] pong!");
  },
});
