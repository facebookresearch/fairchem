import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/elm.js
function switchState(source, setState, f) {
  setState(f);
  return f(source, setState);
}
function normal() {
  return function(source, setState) {
    if (source.eatWhile(spacesRE)) {
      return null;
    }
    var char = source.next();
    if (specialRE.test(char)) {
      return char === "{" && source.eat("-") ? switchState(source, setState, chompMultiComment(1)) : char === "[" && source.match("glsl|") ? switchState(source, setState, chompGlsl) : "builtin";
    }
    if (char === "'") {
      return switchState(source, setState, chompChar);
    }
    if (char === '"') {
      return source.eat('"') ? source.eat('"') ? switchState(source, setState, chompMultiString) : "string" : switchState(source, setState, chompSingleString);
    }
    if (upperRE.test(char)) {
      source.eatWhile(innerRE);
      return "type";
    }
    if (lowerRE.test(char)) {
      var isDef = source.pos === 1;
      source.eatWhile(innerRE);
      return isDef ? "def" : "variable";
    }
    if (digitRE.test(char)) {
      if (char === "0") {
        if (source.eat(/[xX]/)) {
          source.eatWhile(hexRE);
          return "number";
        }
      } else {
        source.eatWhile(digitRE);
      }
      if (source.eat(".")) {
        source.eatWhile(digitRE);
      }
      if (source.eat(/[eE]/)) {
        source.eat(/[-+]/);
        source.eatWhile(digitRE);
      }
      return "number";
    }
    if (symbolRE.test(char)) {
      if (char === "-" && source.eat("-")) {
        source.skipToEnd();
        return "comment";
      }
      source.eatWhile(symbolRE);
      return "keyword";
    }
    if (char === "_") {
      return "keyword";
    }
    return "error";
  };
}
function chompMultiComment(nest) {
  if (nest == 0) {
    return normal();
  }
  return function(source, setState) {
    while (!source.eol()) {
      var char = source.next();
      if (char == "{" && source.eat("-")) {
        ++nest;
      } else if (char == "-" && source.eat("}")) {
        --nest;
        if (nest === 0) {
          setState(normal());
          return "comment";
        }
      }
    }
    setState(chompMultiComment(nest));
    return "comment";
  };
}
function chompMultiString(source, setState) {
  while (!source.eol()) {
    var char = source.next();
    if (char === '"' && source.eat('"') && source.eat('"')) {
      setState(normal());
      return "string";
    }
  }
  return "string";
}
function chompSingleString(source, setState) {
  while (source.skipTo('\\"')) {
    source.next();
    source.next();
  }
  if (source.skipTo('"')) {
    source.next();
    setState(normal());
    return "string";
  }
  source.skipToEnd();
  setState(normal());
  return "error";
}
function chompChar(source, setState) {
  while (source.skipTo("\\'")) {
    source.next();
    source.next();
  }
  if (source.skipTo("'")) {
    source.next();
    setState(normal());
    return "string";
  }
  source.skipToEnd();
  setState(normal());
  return "error";
}
function chompGlsl(source, setState) {
  while (!source.eol()) {
    var char = source.next();
    if (char === "|" && source.eat("]")) {
      setState(normal());
      return "string";
    }
  }
  return "string";
}
var lowerRE, upperRE, innerRE, digitRE, hexRE, symbolRE, specialRE, spacesRE, wellKnownWords, elm;
var init_elm = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/elm.js"() {
    lowerRE = /[a-z]/;
    upperRE = /[A-Z]/;
    innerRE = /[a-zA-Z0-9_]/;
    digitRE = /[0-9]/;
    hexRE = /[0-9A-Fa-f]/;
    symbolRE = /[-&*+.\\/<>=?^|:]/;
    specialRE = /[(),[\]{}]/;
    spacesRE = /[ \v\f]/;
    wellKnownWords = {
      case: 1,
      of: 1,
      as: 1,
      if: 1,
      then: 1,
      else: 1,
      let: 1,
      in: 1,
      type: 1,
      alias: 1,
      module: 1,
      where: 1,
      import: 1,
      exposing: 1,
      port: 1
    };
    elm = {
      name: "elm",
      startState: function() {
        return { f: normal() };
      },
      copyState: function(s) {
        return { f: s.f };
      },
      token: function(stream, state) {
        var type = state.f(stream, function(s) {
          state.f = s;
        });
        var word = stream.current();
        return wellKnownWords.hasOwnProperty(word) ? "keyword" : type;
      },
      languageData: {
        commentTokens: { line: "--" }
      }
    };
  }
});
init_elm();
export {
  elm
};
//# sourceMappingURL=/build/_shared/elm-NKVT3CLC.js.map
