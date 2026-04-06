import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/yaml.js
var cons, keywordRegex, yaml;
var init_yaml = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/yaml.js"() {
    cons = ["true", "false", "on", "off", "yes", "no"];
    keywordRegex = new RegExp("\\b((" + cons.join(")|(") + "))$", "i");
    yaml = {
      name: "yaml",
      token: function(stream, state) {
        var ch = stream.peek();
        var esc = state.escaped;
        state.escaped = false;
        if (ch == "#" && (stream.pos == 0 || /\s/.test(stream.string.charAt(stream.pos - 1)))) {
          stream.skipToEnd();
          return "comment";
        }
        if (stream.match(/^('([^']|\\.)*'?|"([^"]|\\.)*"?)/))
          return "string";
        if (state.literal && stream.indentation() > state.keyCol) {
          stream.skipToEnd();
          return "string";
        } else if (state.literal) {
          state.literal = false;
        }
        if (stream.sol()) {
          state.keyCol = 0;
          state.pair = false;
          state.pairStart = false;
          if (stream.match("---")) {
            return "def";
          }
          if (stream.match("...")) {
            return "def";
          }
          if (stream.match(/^\s*-\s+/)) {
            return "meta";
          }
        }
        if (stream.match(/^(\{|\}|\[|\])/)) {
          if (ch == "{")
            state.inlinePairs++;
          else if (ch == "}")
            state.inlinePairs--;
          else if (ch == "[")
            state.inlineList++;
          else
            state.inlineList--;
          return "meta";
        }
        if (state.inlineList > 0 && !esc && ch == ",") {
          stream.next();
          return "meta";
        }
        if (state.inlinePairs > 0 && !esc && ch == ",") {
          state.keyCol = 0;
          state.pair = false;
          state.pairStart = false;
          stream.next();
          return "meta";
        }
        if (state.pairStart) {
          if (stream.match(/^\s*(\||\>)\s*/)) {
            state.literal = true;
            return "meta";
          }
          ;
          if (stream.match(/^\s*(\&|\*)[a-z0-9\._-]+\b/i)) {
            return "variable";
          }
          if (state.inlinePairs == 0 && stream.match(/^\s*-?[0-9\.\,]+\s?$/)) {
            return "number";
          }
          if (state.inlinePairs > 0 && stream.match(/^\s*-?[0-9\.\,]+\s?(?=(,|}))/)) {
            return "number";
          }
          if (stream.match(keywordRegex)) {
            return "keyword";
          }
        }
        if (!state.pair && stream.match(/^\s*(?:[,\[\]{}&*!|>'"%@`][^\s'":]|[^,\[\]{}#&*!|>'"%@`])[^#]*?(?=\s*:($|\s))/)) {
          state.pair = true;
          state.keyCol = stream.indentation();
          return "atom";
        }
        if (state.pair && stream.match(/^:\s*/)) {
          state.pairStart = true;
          return "meta";
        }
        state.pairStart = false;
        state.escaped = ch == "\\";
        stream.next();
        return null;
      },
      startState: function() {
        return {
          pair: false,
          pairStart: false,
          keyCol: 0,
          inlinePairs: 0,
          inlineList: 0,
          literal: false,
          escaped: false
        };
      },
      languageData: {
        commentTokens: { line: "#" }
      }
    };
  }
});
init_yaml();
export {
  yaml
};
//# sourceMappingURL=/build/_shared/yaml-CAPVMBQ5.js.map
