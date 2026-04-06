import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/troff.js
function tokenBase(stream) {
  if (stream.eatSpace())
    return null;
  var sol = stream.sol();
  var ch = stream.next();
  if (ch === "\\") {
    if (stream.match("fB") || stream.match("fR") || stream.match("fI") || stream.match("u") || stream.match("d") || stream.match("%") || stream.match("&")) {
      return "string";
    }
    if (stream.match("m[")) {
      stream.skipTo("]");
      stream.next();
      return "string";
    }
    if (stream.match("s+") || stream.match("s-")) {
      stream.eatWhile(/[\d-]/);
      return "string";
    }
    if (stream.match("(") || stream.match("*(")) {
      stream.eatWhile(/[\w-]/);
      return "string";
    }
    return "string";
  }
  if (sol && (ch === "." || ch === "'")) {
    if (stream.eat("\\") && stream.eat('"')) {
      stream.skipToEnd();
      return "comment";
    }
  }
  if (sol && ch === ".") {
    if (stream.match("B ") || stream.match("I ") || stream.match("R ")) {
      return "attribute";
    }
    if (stream.match("TH ") || stream.match("SH ") || stream.match("SS ") || stream.match("HP ")) {
      stream.skipToEnd();
      return "quote";
    }
    if (stream.match(/[A-Z]/) && stream.match(/[A-Z]/) || stream.match(/[a-z]/) && stream.match(/[a-z]/)) {
      return "attribute";
    }
  }
  stream.eatWhile(/[\w-]/);
  var cur = stream.current();
  return words.hasOwnProperty(cur) ? words[cur] : null;
}
function tokenize(stream, state) {
  return (state.tokens[0] || tokenBase)(stream, state);
}
var words, troff;
var init_troff = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/troff.js"() {
    words = {};
    troff = {
      name: "troff",
      startState: function() {
        return { tokens: [] };
      },
      token: function(stream, state) {
        return tokenize(stream, state);
      }
    };
  }
});
init_troff();
export {
  troff
};
//# sourceMappingURL=/build/_shared/troff-GQ2GNUXY.js.map
