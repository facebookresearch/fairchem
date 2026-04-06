import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/modelica.js
function words(str) {
  var obj = {}, words2 = str.split(" ");
  for (var i = 0; i < words2.length; ++i)
    obj[words2[i]] = true;
  return obj;
}
function tokenLineComment(stream, state) {
  stream.skipToEnd();
  state.tokenize = null;
  return "comment";
}
function tokenBlockComment(stream, state) {
  var maybeEnd = false, ch;
  while (ch = stream.next()) {
    if (maybeEnd && ch == "/") {
      state.tokenize = null;
      break;
    }
    maybeEnd = ch == "*";
  }
  return "comment";
}
function tokenString(stream, state) {
  var escaped = false, ch;
  while ((ch = stream.next()) != null) {
    if (ch == '"' && !escaped) {
      state.tokenize = null;
      state.sol = false;
      break;
    }
    escaped = !escaped && ch == "\\";
  }
  return "string";
}
function tokenIdent(stream, state) {
  stream.eatWhile(isDigit);
  while (stream.eat(isDigit) || stream.eat(isNonDigit)) {
  }
  var cur = stream.current();
  if (state.sol && (cur == "package" || cur == "model" || cur == "when" || cur == "connector"))
    state.level++;
  else if (state.sol && cur == "end" && state.level > 0)
    state.level--;
  state.tokenize = null;
  state.sol = false;
  if (keywords.propertyIsEnumerable(cur))
    return "keyword";
  else if (builtin.propertyIsEnumerable(cur))
    return "builtin";
  else if (atoms.propertyIsEnumerable(cur))
    return "atom";
  else
    return "variable";
}
function tokenQIdent(stream, state) {
  while (stream.eat(/[^']/)) {
  }
  state.tokenize = null;
  state.sol = false;
  if (stream.eat("'"))
    return "variable";
  else
    return "error";
}
function tokenUnsignedNumber(stream, state) {
  stream.eatWhile(isDigit);
  if (stream.eat(".")) {
    stream.eatWhile(isDigit);
  }
  if (stream.eat("e") || stream.eat("E")) {
    if (!stream.eat("-"))
      stream.eat("+");
    stream.eatWhile(isDigit);
  }
  state.tokenize = null;
  state.sol = false;
  return "number";
}
var keywords, builtin, atoms, completions, isSingleOperatorChar, isDoubleOperatorChar, isDigit, isNonDigit, modelica;
var init_modelica = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/modelica.js"() {
    keywords = words("algorithm and annotation assert block break class connect connector constant constrainedby der discrete each else elseif elsewhen encapsulated end enumeration equation expandable extends external false final flow for function if import impure in initial inner input loop model not operator or outer output package parameter partial protected public pure record redeclare replaceable return stream then true type when while within");
    builtin = words("abs acos actualStream asin atan atan2 cardinality ceil cos cosh delay div edge exp floor getInstanceName homotopy inStream integer log log10 mod pre reinit rem semiLinear sign sin sinh spatialDistribution sqrt tan tanh");
    atoms = words("Real Boolean Integer String");
    completions = [].concat(Object.keys(keywords), Object.keys(builtin), Object.keys(atoms));
    isSingleOperatorChar = /[;=\(:\),{}.*<>+\-\/^\[\]]/;
    isDoubleOperatorChar = /(:=|<=|>=|==|<>|\.\+|\.\-|\.\*|\.\/|\.\^)/;
    isDigit = /[0-9]/;
    isNonDigit = /[_a-zA-Z]/;
    modelica = {
      name: "modelica",
      startState: function() {
        return {
          tokenize: null,
          level: 0,
          sol: true
        };
      },
      token: function(stream, state) {
        if (state.tokenize != null) {
          return state.tokenize(stream, state);
        }
        if (stream.sol()) {
          state.sol = true;
        }
        if (stream.eatSpace()) {
          state.tokenize = null;
          return null;
        }
        var ch = stream.next();
        if (ch == "/" && stream.eat("/")) {
          state.tokenize = tokenLineComment;
        } else if (ch == "/" && stream.eat("*")) {
          state.tokenize = tokenBlockComment;
        } else if (isDoubleOperatorChar.test(ch + stream.peek())) {
          stream.next();
          state.tokenize = null;
          return "operator";
        } else if (isSingleOperatorChar.test(ch)) {
          state.tokenize = null;
          return "operator";
        } else if (isNonDigit.test(ch)) {
          state.tokenize = tokenIdent;
        } else if (ch == "'" && stream.peek() && stream.peek() != "'") {
          state.tokenize = tokenQIdent;
        } else if (ch == '"') {
          state.tokenize = tokenString;
        } else if (isDigit.test(ch)) {
          state.tokenize = tokenUnsignedNumber;
        } else {
          state.tokenize = null;
          return "error";
        }
        return state.tokenize(stream, state);
      },
      indent: function(state, textAfter, cx) {
        if (state.tokenize != null)
          return null;
        var level = state.level;
        if (/(algorithm)/.test(textAfter))
          level--;
        if (/(equation)/.test(textAfter))
          level--;
        if (/(initial algorithm)/.test(textAfter))
          level--;
        if (/(initial equation)/.test(textAfter))
          level--;
        if (/(end)/.test(textAfter))
          level--;
        if (level > 0)
          return cx.unit * level;
        else
          return 0;
      },
      languageData: {
        commentTokens: { line: "//", block: { open: "/*", close: "*/" } },
        autocomplete: completions
      }
    };
  }
});
init_modelica();
export {
  modelica
};
//# sourceMappingURL=/build/_shared/modelica-VSWV7PFR.js.map
