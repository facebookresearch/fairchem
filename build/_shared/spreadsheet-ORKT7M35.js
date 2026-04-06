import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/spreadsheet.js
var spreadsheet;
var init_spreadsheet = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/spreadsheet.js"() {
    spreadsheet = {
      name: "spreadsheet",
      startState: function() {
        return {
          stringType: null,
          stack: []
        };
      },
      token: function(stream, state) {
        if (!stream)
          return;
        if (state.stack.length === 0) {
          if (stream.peek() == '"' || stream.peek() == "'") {
            state.stringType = stream.peek();
            stream.next();
            state.stack.unshift("string");
          }
        }
        switch (state.stack[0]) {
          case "string":
            while (state.stack[0] === "string" && !stream.eol()) {
              if (stream.peek() === state.stringType) {
                stream.next();
                state.stack.shift();
              } else if (stream.peek() === "\\") {
                stream.next();
                stream.next();
              } else {
                stream.match(/^.[^\\\"\']*/);
              }
            }
            return "string";
          case "characterClass":
            while (state.stack[0] === "characterClass" && !stream.eol()) {
              if (!(stream.match(/^[^\]\\]+/) || stream.match(/^\\./)))
                state.stack.shift();
            }
            return "operator";
        }
        var peek = stream.peek();
        switch (peek) {
          case "[":
            stream.next();
            state.stack.unshift("characterClass");
            return "bracket";
          case ":":
            stream.next();
            return "operator";
          case "\\":
            if (stream.match(/\\[a-z]+/))
              return "string.special";
            else {
              stream.next();
              return "atom";
            }
          case ".":
          case ",":
          case ";":
          case "*":
          case "-":
          case "+":
          case "^":
          case "<":
          case "/":
          case "=":
            stream.next();
            return "atom";
          case "$":
            stream.next();
            return "builtin";
        }
        if (stream.match(/\d+/)) {
          if (stream.match(/^\w+/))
            return "error";
          return "number";
        } else if (stream.match(/^[a-zA-Z_]\w*/)) {
          if (stream.match(/(?=[\(.])/, false))
            return "keyword";
          return "variable";
        } else if (["[", "]", "(", ")", "{", "}"].indexOf(peek) != -1) {
          stream.next();
          return "bracket";
        } else if (!stream.eatSpace()) {
          stream.next();
        }
        return null;
      }
    };
  }
});
init_spreadsheet();
export {
  spreadsheet
};
//# sourceMappingURL=/build/_shared/spreadsheet-ORKT7M35.js.map
