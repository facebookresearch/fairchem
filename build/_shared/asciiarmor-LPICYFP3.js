import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/asciiarmor.js
function errorIfNotEmpty(stream) {
  var nonWS = stream.match(/^\s*\S/);
  stream.skipToEnd();
  return nonWS ? "error" : null;
}
var asciiArmor;
var init_asciiarmor = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/asciiarmor.js"() {
    asciiArmor = {
      name: "asciiarmor",
      token: function(stream, state) {
        var m;
        if (state.state == "top") {
          if (stream.sol() && (m = stream.match(/^-----BEGIN (.*)?-----\s*$/))) {
            state.state = "headers";
            state.type = m[1];
            return "tag";
          }
          return errorIfNotEmpty(stream);
        } else if (state.state == "headers") {
          if (stream.sol() && stream.match(/^\w+:/)) {
            state.state = "header";
            return "atom";
          } else {
            var result = errorIfNotEmpty(stream);
            if (result)
              state.state = "body";
            return result;
          }
        } else if (state.state == "header") {
          stream.skipToEnd();
          state.state = "headers";
          return "string";
        } else if (state.state == "body") {
          if (stream.sol() && (m = stream.match(/^-----END (.*)?-----\s*$/))) {
            if (m[1] != state.type)
              return "error";
            state.state = "end";
            return "tag";
          } else {
            if (stream.eatWhile(/[A-Za-z0-9+\/=]/)) {
              return null;
            } else {
              stream.next();
              return "error";
            }
          }
        } else if (state.state == "end") {
          return errorIfNotEmpty(stream);
        }
      },
      blankLine: function(state) {
        if (state.state == "headers")
          state.state = "body";
      },
      startState: function() {
        return { state: "top", type: null };
      }
    };
  }
});
init_asciiarmor();
export {
  asciiArmor
};
//# sourceMappingURL=/build/_shared/asciiarmor-LPICYFP3.js.map
