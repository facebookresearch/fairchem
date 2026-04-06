import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/tiki.js
function inBlock(style, terminator, returnTokenizer) {
  return function(stream, state) {
    while (!stream.eol()) {
      if (stream.match(terminator)) {
        state.tokenize = inText;
        break;
      }
      stream.next();
    }
    if (returnTokenizer)
      state.tokenize = returnTokenizer;
    return style;
  };
}
function inLine(style) {
  return function(stream, state) {
    while (!stream.eol()) {
      stream.next();
    }
    state.tokenize = inText;
    return style;
  };
}
function inText(stream, state) {
  function chain(parser) {
    state.tokenize = parser;
    return parser(stream, state);
  }
  var sol = stream.sol();
  var ch = stream.next();
  switch (ch) {
    case "{":
      stream.eat("/");
      stream.eatSpace();
      stream.eatWhile(/[^\s\u00a0=\"\'\/?(}]/);
      state.tokenize = inPlugin;
      return "tag";
    case "_":
      if (stream.eat("_"))
        return chain(inBlock("strong", "__", inText));
      break;
    case "'":
      if (stream.eat("'"))
        return chain(inBlock("em", "''", inText));
      break;
    case "(":
      if (stream.eat("("))
        return chain(inBlock("link", "))", inText));
      break;
    case "[":
      return chain(inBlock("url", "]", inText));
      break;
    case "|":
      if (stream.eat("|"))
        return chain(inBlock("comment", "||"));
      break;
    case "-":
      if (stream.eat("=")) {
        return chain(inBlock("header string", "=-", inText));
      } else if (stream.eat("-")) {
        return chain(inBlock("error tw-deleted", "--", inText));
      }
      break;
    case "=":
      if (stream.match("=="))
        return chain(inBlock("tw-underline", "===", inText));
      break;
    case ":":
      if (stream.eat(":"))
        return chain(inBlock("comment", "::"));
      break;
    case "^":
      return chain(inBlock("tw-box", "^"));
      break;
    case "~":
      if (stream.match("np~"))
        return chain(inBlock("meta", "~/np~"));
      break;
  }
  if (sol) {
    switch (ch) {
      case "!":
        if (stream.match("!!!!!")) {
          return chain(inLine("header string"));
        } else if (stream.match("!!!!")) {
          return chain(inLine("header string"));
        } else if (stream.match("!!!")) {
          return chain(inLine("header string"));
        } else if (stream.match("!!")) {
          return chain(inLine("header string"));
        } else {
          return chain(inLine("header string"));
        }
        break;
      case "*":
      case "#":
      case "+":
        return chain(inLine("tw-listitem bracket"));
        break;
    }
  }
  return null;
}
function inPlugin(stream, state) {
  var ch = stream.next();
  var peek = stream.peek();
  if (ch == "}") {
    state.tokenize = inText;
    return "tag";
  } else if (ch == "(" || ch == ")") {
    return "bracket";
  } else if (ch == "=") {
    type = "equals";
    if (peek == ">") {
      stream.next();
      peek = stream.peek();
    }
    if (!/[\'\"]/.test(peek)) {
      state.tokenize = inAttributeNoQuote();
    }
    return "operator";
  } else if (/[\'\"]/.test(ch)) {
    state.tokenize = inAttribute(ch);
    return state.tokenize(stream, state);
  } else {
    stream.eatWhile(/[^\s\u00a0=\"\'\/?]/);
    return "keyword";
  }
}
function inAttribute(quote) {
  return function(stream, state) {
    while (!stream.eol()) {
      if (stream.next() == quote) {
        state.tokenize = inPlugin;
        break;
      }
    }
    return "string";
  };
}
function inAttributeNoQuote() {
  return function(stream, state) {
    while (!stream.eol()) {
      var ch = stream.next();
      var peek = stream.peek();
      if (ch == " " || ch == "," || /[ )}]/.test(peek)) {
        state.tokenize = inPlugin;
        break;
      }
    }
    return "string";
  };
}
function pass() {
  for (var i = arguments.length - 1; i >= 0; i--)
    curState.cc.push(arguments[i]);
}
function cont() {
  pass.apply(null, arguments);
  return true;
}
function pushContext(pluginName2, startOfLine) {
  var noIndent = curState.context && curState.context.noIndent;
  curState.context = {
    prev: curState.context,
    pluginName: pluginName2,
    indent: curState.indented,
    startOfLine,
    noIndent
  };
}
function popContext() {
  if (curState.context)
    curState.context = curState.context.prev;
}
function element(type2) {
  if (type2 == "openPlugin") {
    curState.pluginName = pluginName;
    return cont(attributes, endplugin(curState.startOfLine));
  } else if (type2 == "closePlugin") {
    var err = false;
    if (curState.context) {
      err = curState.context.pluginName != pluginName;
      popContext();
    } else {
      err = true;
    }
    if (err)
      setStyle = "error";
    return cont(endcloseplugin(err));
  } else if (type2 == "string") {
    if (!curState.context || curState.context.name != "!cdata")
      pushContext("!cdata");
    if (curState.tokenize == inText)
      popContext();
    return cont();
  } else
    return cont();
}
function endplugin(startOfLine) {
  return function(type2) {
    if (type2 == "selfclosePlugin" || type2 == "endPlugin")
      return cont();
    if (type2 == "endPlugin") {
      pushContext(curState.pluginName, startOfLine);
      return cont();
    }
    return cont();
  };
}
function endcloseplugin(err) {
  return function(type2) {
    if (err)
      setStyle = "error";
    if (type2 == "endPlugin")
      return cont();
    return pass();
  };
}
function attributes(type2) {
  if (type2 == "keyword") {
    setStyle = "attribute";
    return cont(attributes);
  }
  if (type2 == "equals")
    return cont(attvalue, attributes);
  return pass();
}
function attvalue(type2) {
  if (type2 == "keyword") {
    setStyle = "string";
    return cont();
  }
  if (type2 == "string")
    return cont(attvaluemaybe);
  return pass();
}
function attvaluemaybe(type2) {
  if (type2 == "string")
    return cont(attvaluemaybe);
  else
    return pass();
}
var pluginName, type, curState, setStyle, tiki;
var init_tiki = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/tiki.js"() {
    tiki = {
      name: "tiki",
      startState: function() {
        return { tokenize: inText, cc: [], indented: 0, startOfLine: true, pluginName: null, context: null };
      },
      token: function(stream, state) {
        if (stream.sol()) {
          state.startOfLine = true;
          state.indented = stream.indentation();
        }
        if (stream.eatSpace())
          return null;
        setStyle = type = pluginName = null;
        var style = state.tokenize(stream, state);
        if ((style || type) && style != "comment") {
          curState = state;
          while (true) {
            var comb = state.cc.pop() || element;
            if (comb(type || style))
              break;
          }
        }
        state.startOfLine = false;
        return setStyle || style;
      },
      indent: function(state, textAfter, cx) {
        var context = state.context;
        if (context && context.noIndent)
          return 0;
        if (context && /^{\//.test(textAfter))
          context = context.prev;
        while (context && !context.startOfLine)
          context = context.prev;
        if (context)
          return context.indent + cx.unit;
        else
          return 0;
      }
    };
  }
});
init_tiki();
export {
  tiki
};
//# sourceMappingURL=/build/_shared/tiki-TW5XTX3P.js.map
