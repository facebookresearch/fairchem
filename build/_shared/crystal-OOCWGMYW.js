import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/crystal.js
function wordRegExp(words, end) {
  return new RegExp((end ? "" : "^") + "(?:" + words.join("|") + ")" + (end ? "$" : "\\b"));
}
function chain(tokenize, stream, state) {
  state.tokenize.push(tokenize);
  return tokenize(stream, state);
}
function tokenBase(stream, state) {
  if (stream.eatSpace()) {
    return null;
  }
  if (state.lastToken != "\\" && stream.match("{%", false)) {
    return chain(tokenMacro("%", "%"), stream, state);
  }
  if (state.lastToken != "\\" && stream.match("{{", false)) {
    return chain(tokenMacro("{", "}"), stream, state);
  }
  if (stream.peek() == "#") {
    stream.skipToEnd();
    return "comment";
  }
  var matched;
  if (stream.match(idents)) {
    stream.eat(/[?!]/);
    matched = stream.current();
    if (stream.eat(":")) {
      return "atom";
    } else if (state.lastToken == ".") {
      return "property";
    } else if (keywords.test(matched)) {
      if (indentKeywords.test(matched)) {
        if (!(matched == "fun" && state.blocks.indexOf("lib") >= 0) && !(matched == "def" && state.lastToken == "abstract")) {
          state.blocks.push(matched);
          state.currentIndent += 1;
        }
      } else if ((state.lastStyle == "operator" || !state.lastStyle) && indentExpressionKeywords.test(matched)) {
        state.blocks.push(matched);
        state.currentIndent += 1;
      } else if (matched == "end") {
        state.blocks.pop();
        state.currentIndent -= 1;
      }
      if (nextTokenizer.hasOwnProperty(matched)) {
        state.tokenize.push(nextTokenizer[matched]);
      }
      return "keyword";
    } else if (atomWords.test(matched)) {
      return "atom";
    }
    return "variable";
  }
  if (stream.eat("@")) {
    if (stream.peek() == "[") {
      return chain(tokenNest("[", "]", "meta"), stream, state);
    }
    stream.eat("@");
    stream.match(idents) || stream.match(types);
    return "propertyName";
  }
  if (stream.match(types)) {
    return "tag";
  }
  if (stream.eat(":")) {
    if (stream.eat('"')) {
      return chain(tokenQuote('"', "atom", false), stream, state);
    } else if (stream.match(idents) || stream.match(types) || stream.match(operators) || stream.match(conditionalOperators) || stream.match(indexingOperators)) {
      return "atom";
    }
    stream.eat(":");
    return "operator";
  }
  if (stream.eat('"')) {
    return chain(tokenQuote('"', "string", true), stream, state);
  }
  if (stream.peek() == "%") {
    var style = "string";
    var embed = true;
    var delim;
    if (stream.match("%r")) {
      style = "string.special";
      delim = stream.next();
    } else if (stream.match("%w")) {
      embed = false;
      delim = stream.next();
    } else if (stream.match("%q")) {
      embed = false;
      delim = stream.next();
    } else {
      if (delim = stream.match(/^%([^\w\s=])/)) {
        delim = delim[1];
      } else if (stream.match(/^%[a-zA-Z_\u009F-\uFFFF][\w\u009F-\uFFFF]*/)) {
        return "meta";
      } else if (stream.eat("%")) {
        return "operator";
      }
    }
    if (matching.hasOwnProperty(delim)) {
      delim = matching[delim];
    }
    return chain(tokenQuote(delim, style, embed), stream, state);
  }
  if (matched = stream.match(/^<<-('?)([A-Z]\w*)\1/)) {
    return chain(tokenHereDoc(matched[2], !matched[1]), stream, state);
  }
  if (stream.eat("'")) {
    stream.match(/^(?:[^']|\\(?:[befnrtv0'"]|[0-7]{3}|u(?:[0-9a-fA-F]{4}|\{[0-9a-fA-F]{1,6}\})))/);
    stream.eat("'");
    return "atom";
  }
  if (stream.eat("0")) {
    if (stream.eat("x")) {
      stream.match(/^[0-9a-fA-F_]+/);
    } else if (stream.eat("o")) {
      stream.match(/^[0-7_]+/);
    } else if (stream.eat("b")) {
      stream.match(/^[01_]+/);
    }
    return "number";
  }
  if (stream.eat(/^\d/)) {
    stream.match(/^[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?/);
    return "number";
  }
  if (stream.match(operators)) {
    stream.eat("=");
    return "operator";
  }
  if (stream.match(conditionalOperators) || stream.match(anotherOperators)) {
    return "operator";
  }
  if (matched = stream.match(/[({[]/, false)) {
    matched = matched[0];
    return chain(tokenNest(matched, matching[matched], null), stream, state);
  }
  if (stream.eat("\\")) {
    stream.next();
    return "meta";
  }
  stream.next();
  return null;
}
function tokenNest(begin, end, style, started) {
  return function(stream, state) {
    if (!started && stream.match(begin)) {
      state.tokenize[state.tokenize.length - 1] = tokenNest(begin, end, style, true);
      state.currentIndent += 1;
      return style;
    }
    var nextStyle = tokenBase(stream, state);
    if (stream.current() === end) {
      state.tokenize.pop();
      state.currentIndent -= 1;
      nextStyle = style;
    }
    return nextStyle;
  };
}
function tokenMacro(begin, end, started) {
  return function(stream, state) {
    if (!started && stream.match("{" + begin)) {
      state.currentIndent += 1;
      state.tokenize[state.tokenize.length - 1] = tokenMacro(begin, end, true);
      return "meta";
    }
    if (stream.match(end + "}")) {
      state.currentIndent -= 1;
      state.tokenize.pop();
      return "meta";
    }
    return tokenBase(stream, state);
  };
}
function tokenMacroDef(stream, state) {
  if (stream.eatSpace()) {
    return null;
  }
  var matched;
  if (matched = stream.match(idents)) {
    if (matched == "def") {
      return "keyword";
    }
    stream.eat(/[?!]/);
  }
  state.tokenize.pop();
  return "def";
}
function tokenFollowIdent(stream, state) {
  if (stream.eatSpace()) {
    return null;
  }
  if (stream.match(idents)) {
    stream.eat(/[!?]/);
  } else {
    stream.match(operators) || stream.match(conditionalOperators) || stream.match(indexingOperators);
  }
  state.tokenize.pop();
  return "def";
}
function tokenFollowType(stream, state) {
  if (stream.eatSpace()) {
    return null;
  }
  stream.match(types);
  state.tokenize.pop();
  return "def";
}
function tokenQuote(end, style, embed) {
  return function(stream, state) {
    var escaped = false;
    while (stream.peek()) {
      if (!escaped) {
        if (stream.match("{%", false)) {
          state.tokenize.push(tokenMacro("%", "%"));
          return style;
        }
        if (stream.match("{{", false)) {
          state.tokenize.push(tokenMacro("{", "}"));
          return style;
        }
        if (embed && stream.match("#{", false)) {
          state.tokenize.push(tokenNest("#{", "}", "meta"));
          return style;
        }
        var ch = stream.next();
        if (ch == end) {
          state.tokenize.pop();
          return style;
        }
        escaped = embed && ch == "\\";
      } else {
        stream.next();
        escaped = false;
      }
    }
    return style;
  };
}
function tokenHereDoc(phrase, embed) {
  return function(stream, state) {
    if (stream.sol()) {
      stream.eatSpace();
      if (stream.match(phrase)) {
        state.tokenize.pop();
        return "string";
      }
    }
    var escaped = false;
    while (stream.peek()) {
      if (!escaped) {
        if (stream.match("{%", false)) {
          state.tokenize.push(tokenMacro("%", "%"));
          return "string";
        }
        if (stream.match("{{", false)) {
          state.tokenize.push(tokenMacro("{", "}"));
          return "string";
        }
        if (embed && stream.match("#{", false)) {
          state.tokenize.push(tokenNest("#{", "}", "meta"));
          return "string";
        }
        escaped = stream.next() == "\\" && embed;
      } else {
        stream.next();
        escaped = false;
      }
    }
    return "string";
  };
}
var operators, conditionalOperators, indexingOperators, anotherOperators, idents, types, keywords, atomWords, indentKeywordsArray, indentKeywords, indentExpressionKeywordsArray, indentExpressionKeywords, dedentKeywordsArray, dedentKeywords, dedentPunctualsArray, dedentPunctuals, nextTokenizer, matching, crystal;
var init_crystal = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/crystal.js"() {
    operators = /^(?:[-+/%|&^]|\*\*?|[<>]{2})/;
    conditionalOperators = /^(?:[=!]~|===|<=>|[<>=!]=?|[|&]{2}|~)/;
    indexingOperators = /^(?:\[\][?=]?)/;
    anotherOperators = /^(?:\.(?:\.{2})?|->|[?:])/;
    idents = /^[a-z_\u009F-\uFFFF][a-zA-Z0-9_\u009F-\uFFFF]*/;
    types = /^[A-Z_\u009F-\uFFFF][a-zA-Z0-9_\u009F-\uFFFF]*/;
    keywords = wordRegExp([
      "abstract",
      "alias",
      "as",
      "asm",
      "begin",
      "break",
      "case",
      "class",
      "def",
      "do",
      "else",
      "elsif",
      "end",
      "ensure",
      "enum",
      "extend",
      "for",
      "fun",
      "if",
      "include",
      "instance_sizeof",
      "lib",
      "macro",
      "module",
      "next",
      "of",
      "out",
      "pointerof",
      "private",
      "protected",
      "rescue",
      "return",
      "require",
      "select",
      "sizeof",
      "struct",
      "super",
      "then",
      "type",
      "typeof",
      "uninitialized",
      "union",
      "unless",
      "until",
      "when",
      "while",
      "with",
      "yield",
      "__DIR__",
      "__END_LINE__",
      "__FILE__",
      "__LINE__"
    ]);
    atomWords = wordRegExp(["true", "false", "nil", "self"]);
    indentKeywordsArray = [
      "def",
      "fun",
      "macro",
      "class",
      "module",
      "struct",
      "lib",
      "enum",
      "union",
      "do",
      "for"
    ];
    indentKeywords = wordRegExp(indentKeywordsArray);
    indentExpressionKeywordsArray = ["if", "unless", "case", "while", "until", "begin", "then"];
    indentExpressionKeywords = wordRegExp(indentExpressionKeywordsArray);
    dedentKeywordsArray = ["end", "else", "elsif", "rescue", "ensure"];
    dedentKeywords = wordRegExp(dedentKeywordsArray);
    dedentPunctualsArray = ["\\)", "\\}", "\\]"];
    dedentPunctuals = new RegExp("^(?:" + dedentPunctualsArray.join("|") + ")$");
    nextTokenizer = {
      "def": tokenFollowIdent,
      "fun": tokenFollowIdent,
      "macro": tokenMacroDef,
      "class": tokenFollowType,
      "module": tokenFollowType,
      "struct": tokenFollowType,
      "lib": tokenFollowType,
      "enum": tokenFollowType,
      "union": tokenFollowType
    };
    matching = { "[": "]", "{": "}", "(": ")", "<": ">" };
    crystal = {
      name: "crystal",
      startState: function() {
        return {
          tokenize: [tokenBase],
          currentIndent: 0,
          lastToken: null,
          lastStyle: null,
          blocks: []
        };
      },
      token: function(stream, state) {
        var style = state.tokenize[state.tokenize.length - 1](stream, state);
        var token = stream.current();
        if (style && style != "comment") {
          state.lastToken = token;
          state.lastStyle = style;
        }
        return style;
      },
      indent: function(state, textAfter, cx) {
        textAfter = textAfter.replace(/^\s*(?:\{%)?\s*|\s*(?:%\})?\s*$/g, "");
        if (dedentKeywords.test(textAfter) || dedentPunctuals.test(textAfter)) {
          return cx.unit * (state.currentIndent - 1);
        }
        return cx.unit * state.currentIndent;
      },
      languageData: {
        indentOnInput: wordRegExp(dedentPunctualsArray.concat(dedentKeywordsArray), true),
        commentTokens: { line: "#" }
      }
    };
  }
});
init_crystal();
export {
  crystal
};
//# sourceMappingURL=/build/_shared/crystal-OOCWGMYW.js.map
