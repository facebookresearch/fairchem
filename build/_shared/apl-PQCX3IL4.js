import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/apl.js
var builtInFuncs, isOperator, isNiladic, isFunction, isArrow, isComment, stringEater, apl;
var init_apl = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/apl.js"() {
    builtInFuncs = {
      "+": ["conjugate", "add"],
      "\u2212": ["negate", "subtract"],
      "\xD7": ["signOf", "multiply"],
      "\xF7": ["reciprocal", "divide"],
      "\u2308": ["ceiling", "greaterOf"],
      "\u230A": ["floor", "lesserOf"],
      "\u2223": ["absolute", "residue"],
      "\u2373": ["indexGenerate", "indexOf"],
      "?": ["roll", "deal"],
      "\u22C6": ["exponentiate", "toThePowerOf"],
      "\u235F": ["naturalLog", "logToTheBase"],
      "\u25CB": ["piTimes", "circularFuncs"],
      "!": ["factorial", "binomial"],
      "\u2339": ["matrixInverse", "matrixDivide"],
      "<": [null, "lessThan"],
      "\u2264": [null, "lessThanOrEqual"],
      "=": [null, "equals"],
      ">": [null, "greaterThan"],
      "\u2265": [null, "greaterThanOrEqual"],
      "\u2260": [null, "notEqual"],
      "\u2261": ["depth", "match"],
      "\u2262": [null, "notMatch"],
      "\u2208": ["enlist", "membership"],
      "\u2377": [null, "find"],
      "\u222A": ["unique", "union"],
      "\u2229": [null, "intersection"],
      "\u223C": ["not", "without"],
      "\u2228": [null, "or"],
      "\u2227": [null, "and"],
      "\u2371": [null, "nor"],
      "\u2372": [null, "nand"],
      "\u2374": ["shapeOf", "reshape"],
      ",": ["ravel", "catenate"],
      "\u236A": [null, "firstAxisCatenate"],
      "\u233D": ["reverse", "rotate"],
      "\u2296": ["axis1Reverse", "axis1Rotate"],
      "\u2349": ["transpose", null],
      "\u2191": ["first", "take"],
      "\u2193": [null, "drop"],
      "\u2282": ["enclose", "partitionWithAxis"],
      "\u2283": ["diclose", "pick"],
      "\u2337": [null, "index"],
      "\u234B": ["gradeUp", null],
      "\u2352": ["gradeDown", null],
      "\u22A4": ["encode", null],
      "\u22A5": ["decode", null],
      "\u2355": ["format", "formatByExample"],
      "\u234E": ["execute", null],
      "\u22A3": ["stop", "left"],
      "\u22A2": ["pass", "right"]
    };
    isOperator = /[\.\/⌿⍀¨⍣]/;
    isNiladic = /⍬/;
    isFunction = /[\+−×÷⌈⌊∣⍳\?⋆⍟○!⌹<≤=>≥≠≡≢∈⍷∪∩∼∨∧⍱⍲⍴,⍪⌽⊖⍉↑↓⊂⊃⌷⍋⍒⊤⊥⍕⍎⊣⊢]/;
    isArrow = /←/;
    isComment = /[⍝#].*$/;
    stringEater = function(type) {
      var prev;
      prev = false;
      return function(c) {
        prev = c;
        if (c === type) {
          return prev === "\\";
        }
        return true;
      };
    };
    apl = {
      name: "apl",
      startState: function() {
        return {
          prev: false,
          func: false,
          op: false,
          string: false,
          escape: false
        };
      },
      token: function(stream, state) {
        var ch;
        if (stream.eatSpace()) {
          return null;
        }
        ch = stream.next();
        if (ch === '"' || ch === "'") {
          stream.eatWhile(stringEater(ch));
          stream.next();
          state.prev = true;
          return "string";
        }
        if (/[\[{\(]/.test(ch)) {
          state.prev = false;
          return null;
        }
        if (/[\]}\)]/.test(ch)) {
          state.prev = true;
          return null;
        }
        if (isNiladic.test(ch)) {
          state.prev = false;
          return "atom";
        }
        if (/[¯\d]/.test(ch)) {
          if (state.func) {
            state.func = false;
            state.prev = false;
          } else {
            state.prev = true;
          }
          stream.eatWhile(/[\w\.]/);
          return "number";
        }
        if (isOperator.test(ch)) {
          return "operator";
        }
        if (isArrow.test(ch)) {
          return "operator";
        }
        if (isFunction.test(ch)) {
          state.func = true;
          state.prev = false;
          return builtInFuncs[ch] ? "variableName.function.standard" : "variableName.function";
        }
        if (isComment.test(ch)) {
          stream.skipToEnd();
          return "comment";
        }
        if (ch === "\u2218" && stream.peek() === ".") {
          stream.next();
          return "variableName.function";
        }
        stream.eatWhile(/[\w\$_]/);
        state.prev = true;
        return "keyword";
      }
    };
  }
});
init_apl();
export {
  apl
};
//# sourceMappingURL=/build/_shared/apl-PQCX3IL4.js.map
