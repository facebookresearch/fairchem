import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/vb.js
function wordRegexp(words) {
  return new RegExp("^((" + words.join(")|(") + "))\\b", "i");
}
function indent(_stream, state) {
  state.currentIndent++;
}
function dedent(_stream, state) {
  state.currentIndent--;
}
function tokenBase(stream, state) {
  if (stream.eatSpace()) {
    return null;
  }
  var ch = stream.peek();
  if (ch === "'") {
    stream.skipToEnd();
    return "comment";
  }
  if (stream.match(/^((&H)|(&O))?[0-9\.a-f]/i, false)) {
    var floatLiteral = false;
    if (stream.match(/^\d*\.\d+F?/i)) {
      floatLiteral = true;
    } else if (stream.match(/^\d+\.\d*F?/)) {
      floatLiteral = true;
    } else if (stream.match(/^\.\d+F?/)) {
      floatLiteral = true;
    }
    if (floatLiteral) {
      stream.eat(/J/i);
      return "number";
    }
    var intLiteral = false;
    if (stream.match(/^&H[0-9a-f]+/i)) {
      intLiteral = true;
    } else if (stream.match(/^&O[0-7]+/i)) {
      intLiteral = true;
    } else if (stream.match(/^[1-9]\d*F?/)) {
      stream.eat(/J/i);
      intLiteral = true;
    } else if (stream.match(/^0(?![\dx])/i)) {
      intLiteral = true;
    }
    if (intLiteral) {
      stream.eat(/L/i);
      return "number";
    }
  }
  if (stream.match(stringPrefixes)) {
    state.tokenize = tokenStringFactory(stream.current());
    return state.tokenize(stream, state);
  }
  if (stream.match(tripleDelimiters) || stream.match(doubleDelimiters)) {
    return null;
  }
  if (stream.match(doubleOperators) || stream.match(singleOperators) || stream.match(wordOperators)) {
    return "operator";
  }
  if (stream.match(singleDelimiters)) {
    return null;
  }
  if (stream.match(doOpening)) {
    indent(stream, state);
    state.doInCurrentLine = true;
    return "keyword";
  }
  if (stream.match(opening)) {
    if (!state.doInCurrentLine)
      indent(stream, state);
    else
      state.doInCurrentLine = false;
    return "keyword";
  }
  if (stream.match(middle)) {
    return "keyword";
  }
  if (stream.match(doubleClosing)) {
    dedent(stream, state);
    dedent(stream, state);
    return "keyword";
  }
  if (stream.match(closing)) {
    dedent(stream, state);
    return "keyword";
  }
  if (stream.match(types)) {
    return "keyword";
  }
  if (stream.match(keywords)) {
    return "keyword";
  }
  if (stream.match(identifiers)) {
    return "variable";
  }
  stream.next();
  return ERRORCLASS;
}
function tokenStringFactory(delimiter) {
  var singleline = delimiter.length == 1;
  var OUTCLASS = "string";
  return function(stream, state) {
    while (!stream.eol()) {
      stream.eatWhile(/[^'"]/);
      if (stream.match(delimiter)) {
        state.tokenize = tokenBase;
        return OUTCLASS;
      } else {
        stream.eat(/['"]/);
      }
    }
    if (singleline) {
      state.tokenize = tokenBase;
    }
    return OUTCLASS;
  };
}
function tokenLexer(stream, state) {
  var style = state.tokenize(stream, state);
  var current = stream.current();
  if (current === ".") {
    style = state.tokenize(stream, state);
    if (style === "variable") {
      return "variable";
    } else {
      return ERRORCLASS;
    }
  }
  var delimiter_index = "[({".indexOf(current);
  if (delimiter_index !== -1) {
    indent(stream, state);
  }
  if (indentInfo === "dedent") {
    if (dedent(stream, state)) {
      return ERRORCLASS;
    }
  }
  delimiter_index = "])}".indexOf(current);
  if (delimiter_index !== -1) {
    if (dedent(stream, state)) {
      return ERRORCLASS;
    }
  }
  return style;
}
var ERRORCLASS, singleOperators, singleDelimiters, doubleOperators, doubleDelimiters, tripleDelimiters, identifiers, openingKeywords, middleKeywords, endKeywords, operatorKeywords, wordOperators, commonKeywords, commontypes, keywords, types, stringPrefixes, opening, middle, closing, doubleClosing, doOpening, indentInfo, vb;
var init_vb = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/vb.js"() {
    ERRORCLASS = "error";
    singleOperators = new RegExp("^[\\+\\-\\*/%&\\\\|\\^~<>!]");
    singleDelimiters = new RegExp("^[\\(\\)\\[\\]\\{\\}@,:`=;\\.]");
    doubleOperators = new RegExp("^((==)|(<>)|(<=)|(>=)|(<>)|(<<)|(>>)|(//)|(\\*\\*))");
    doubleDelimiters = new RegExp("^((\\+=)|(\\-=)|(\\*=)|(%=)|(/=)|(&=)|(\\|=)|(\\^=))");
    tripleDelimiters = new RegExp("^((//=)|(>>=)|(<<=)|(\\*\\*=))");
    identifiers = new RegExp("^[_A-Za-z][_A-Za-z0-9]*");
    openingKeywords = ["class", "module", "sub", "enum", "select", "while", "if", "function", "get", "set", "property", "try", "structure", "synclock", "using", "with"];
    middleKeywords = ["else", "elseif", "case", "catch", "finally"];
    endKeywords = ["next", "loop"];
    operatorKeywords = ["and", "andalso", "or", "orelse", "xor", "in", "not", "is", "isnot", "like"];
    wordOperators = wordRegexp(operatorKeywords);
    commonKeywords = ["#const", "#else", "#elseif", "#end", "#if", "#region", "addhandler", "addressof", "alias", "as", "byref", "byval", "cbool", "cbyte", "cchar", "cdate", "cdbl", "cdec", "cint", "clng", "cobj", "compare", "const", "continue", "csbyte", "cshort", "csng", "cstr", "cuint", "culng", "cushort", "declare", "default", "delegate", "dim", "directcast", "each", "erase", "error", "event", "exit", "explicit", "false", "for", "friend", "gettype", "goto", "handles", "implements", "imports", "infer", "inherits", "interface", "isfalse", "istrue", "lib", "me", "mod", "mustinherit", "mustoverride", "my", "mybase", "myclass", "namespace", "narrowing", "new", "nothing", "notinheritable", "notoverridable", "of", "off", "on", "operator", "option", "optional", "out", "overloads", "overridable", "overrides", "paramarray", "partial", "private", "protected", "public", "raiseevent", "readonly", "redim", "removehandler", "resume", "return", "shadows", "shared", "static", "step", "stop", "strict", "then", "throw", "to", "true", "trycast", "typeof", "until", "until", "when", "widening", "withevents", "writeonly"];
    commontypes = ["object", "boolean", "char", "string", "byte", "sbyte", "short", "ushort", "int16", "uint16", "integer", "uinteger", "int32", "uint32", "long", "ulong", "int64", "uint64", "decimal", "single", "double", "float", "date", "datetime", "intptr", "uintptr"];
    keywords = wordRegexp(commonKeywords);
    types = wordRegexp(commontypes);
    stringPrefixes = '"';
    opening = wordRegexp(openingKeywords);
    middle = wordRegexp(middleKeywords);
    closing = wordRegexp(endKeywords);
    doubleClosing = wordRegexp(["end"]);
    doOpening = wordRegexp(["do"]);
    indentInfo = null;
    vb = {
      name: "vb",
      startState: function() {
        return {
          tokenize: tokenBase,
          lastToken: null,
          currentIndent: 0,
          nextLineIndent: 0,
          doInCurrentLine: false
        };
      },
      token: function(stream, state) {
        if (stream.sol()) {
          state.currentIndent += state.nextLineIndent;
          state.nextLineIndent = 0;
          state.doInCurrentLine = 0;
        }
        var style = tokenLexer(stream, state);
        state.lastToken = { style, content: stream.current() };
        return style;
      },
      indent: function(state, textAfter, cx) {
        var trueText = textAfter.replace(/^\s+|\s+$/g, "");
        if (trueText.match(closing) || trueText.match(doubleClosing) || trueText.match(middle))
          return cx.unit * (state.currentIndent - 1);
        if (state.currentIndent < 0)
          return 0;
        return state.currentIndent * cx.unit;
      },
      languageData: {
        closeBrackets: { brackets: ["(", "[", "{", '"'] },
        commentTokens: { line: "'" },
        autocomplete: openingKeywords.concat(middleKeywords).concat(endKeywords).concat(operatorKeywords).concat(commonKeywords).concat(commontypes)
      }
    };
  }
});
init_vb();
export {
  vb
};
//# sourceMappingURL=/build/_shared/vb-LMC6HBA2.js.map
