import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/css.js
function mkCSS(parserConfig) {
  parserConfig = { ...defaults, ...parserConfig };
  var inline = parserConfig.inline;
  var tokenHooks = parserConfig.tokenHooks, documentTypes2 = parserConfig.documentTypes || {}, mediaTypes2 = parserConfig.mediaTypes || {}, mediaFeatures2 = parserConfig.mediaFeatures || {}, mediaValueKeywords2 = parserConfig.mediaValueKeywords || {}, propertyKeywords2 = parserConfig.propertyKeywords || {}, nonStandardPropertyKeywords2 = parserConfig.nonStandardPropertyKeywords || {}, fontProperties2 = parserConfig.fontProperties || {}, counterDescriptors2 = parserConfig.counterDescriptors || {}, colorKeywords2 = parserConfig.colorKeywords || {}, valueKeywords2 = parserConfig.valueKeywords || {}, allowNested = parserConfig.allowNested, lineComment = parserConfig.lineComment, supportsAtComponent = parserConfig.supportsAtComponent === true, highlightNonStandardPropertyKeywords = parserConfig.highlightNonStandardPropertyKeywords !== false;
  var type, override;
  function ret(style, tp) {
    type = tp;
    return style;
  }
  function tokenBase(stream, state) {
    var ch = stream.next();
    if (tokenHooks[ch]) {
      var result = tokenHooks[ch](stream, state);
      if (result !== false)
        return result;
    }
    if (ch == "@") {
      stream.eatWhile(/[\w\\\-]/);
      return ret("def", stream.current());
    } else if (ch == "=" || (ch == "~" || ch == "|") && stream.eat("=")) {
      return ret(null, "compare");
    } else if (ch == '"' || ch == "'") {
      state.tokenize = tokenString(ch);
      return state.tokenize(stream, state);
    } else if (ch == "#") {
      stream.eatWhile(/[\w\\\-]/);
      return ret("atom", "hash");
    } else if (ch == "!") {
      stream.match(/^\s*\w*/);
      return ret("keyword", "important");
    } else if (/\d/.test(ch) || ch == "." && stream.eat(/\d/)) {
      stream.eatWhile(/[\w.%]/);
      return ret("number", "unit");
    } else if (ch === "-") {
      if (/[\d.]/.test(stream.peek())) {
        stream.eatWhile(/[\w.%]/);
        return ret("number", "unit");
      } else if (stream.match(/^-[\w\\\-]*/)) {
        stream.eatWhile(/[\w\\\-]/);
        if (stream.match(/^\s*:/, false))
          return ret("def", "variable-definition");
        return ret("variableName", "variable");
      } else if (stream.match(/^\w+-/)) {
        return ret("meta", "meta");
      }
    } else if (/[,+>*\/]/.test(ch)) {
      return ret(null, "select-op");
    } else if (ch == "." && stream.match(/^-?[_a-z][_a-z0-9-]*/i)) {
      return ret("qualifier", "qualifier");
    } else if (/[:;{}\[\]\(\)]/.test(ch)) {
      return ret(null, ch);
    } else if (stream.match(/^[\w-.]+(?=\()/)) {
      if (/^(url(-prefix)?|domain|regexp)$/i.test(stream.current())) {
        state.tokenize = tokenParenthesized;
      }
      return ret("variableName.function", "variable");
    } else if (/[\w\\\-]/.test(ch)) {
      stream.eatWhile(/[\w\\\-]/);
      return ret("property", "word");
    } else {
      return ret(null, null);
    }
  }
  function tokenString(quote) {
    return function(stream, state) {
      var escaped = false, ch;
      while ((ch = stream.next()) != null) {
        if (ch == quote && !escaped) {
          if (quote == ")")
            stream.backUp(1);
          break;
        }
        escaped = !escaped && ch == "\\";
      }
      if (ch == quote || !escaped && quote != ")")
        state.tokenize = null;
      return ret("string", "string");
    };
  }
  function tokenParenthesized(stream, state) {
    stream.next();
    if (!stream.match(/^\s*[\"\')]/, false))
      state.tokenize = tokenString(")");
    else
      state.tokenize = null;
    return ret(null, "(");
  }
  function Context(type2, indent, prev) {
    this.type = type2;
    this.indent = indent;
    this.prev = prev;
  }
  function pushContext(state, stream, type2, indent) {
    state.context = new Context(type2, stream.indentation() + (indent === false ? 0 : stream.indentUnit), state.context);
    return type2;
  }
  function popContext(state) {
    if (state.context.prev)
      state.context = state.context.prev;
    return state.context.type;
  }
  function pass(type2, stream, state) {
    return states[state.context.type](type2, stream, state);
  }
  function popAndPass(type2, stream, state, n) {
    for (var i = n || 1; i > 0; i--)
      state.context = state.context.prev;
    return pass(type2, stream, state);
  }
  function wordAsValue(stream) {
    var word = stream.current().toLowerCase();
    if (valueKeywords2.hasOwnProperty(word))
      override = "atom";
    else if (colorKeywords2.hasOwnProperty(word))
      override = "keyword";
    else
      override = "variable";
  }
  var states = {};
  states.top = function(type2, stream, state) {
    if (type2 == "{") {
      return pushContext(state, stream, "block");
    } else if (type2 == "}" && state.context.prev) {
      return popContext(state);
    } else if (supportsAtComponent && /@component/i.test(type2)) {
      return pushContext(state, stream, "atComponentBlock");
    } else if (/^@(-moz-)?document$/i.test(type2)) {
      return pushContext(state, stream, "documentTypes");
    } else if (/^@(media|supports|(-moz-)?document|import)$/i.test(type2)) {
      return pushContext(state, stream, "atBlock");
    } else if (/^@(font-face|counter-style)/i.test(type2)) {
      state.stateArg = type2;
      return "restricted_atBlock_before";
    } else if (/^@(-(moz|ms|o|webkit)-)?keyframes$/i.test(type2)) {
      return "keyframes";
    } else if (type2 && type2.charAt(0) == "@") {
      return pushContext(state, stream, "at");
    } else if (type2 == "hash") {
      override = "builtin";
    } else if (type2 == "word") {
      override = "tag";
    } else if (type2 == "variable-definition") {
      return "maybeprop";
    } else if (type2 == "interpolation") {
      return pushContext(state, stream, "interpolation");
    } else if (type2 == ":") {
      return "pseudo";
    } else if (allowNested && type2 == "(") {
      return pushContext(state, stream, "parens");
    }
    return state.context.type;
  };
  states.block = function(type2, stream, state) {
    if (type2 == "word") {
      var word = stream.current().toLowerCase();
      if (propertyKeywords2.hasOwnProperty(word)) {
        override = "property";
        return "maybeprop";
      } else if (nonStandardPropertyKeywords2.hasOwnProperty(word)) {
        override = highlightNonStandardPropertyKeywords ? "string.special" : "property";
        return "maybeprop";
      } else if (allowNested) {
        override = stream.match(/^\s*:(?:\s|$)/, false) ? "property" : "tag";
        return "block";
      } else {
        override = "error";
        return "maybeprop";
      }
    } else if (type2 == "meta") {
      return "block";
    } else if (!allowNested && (type2 == "hash" || type2 == "qualifier")) {
      override = "error";
      return "block";
    } else {
      return states.top(type2, stream, state);
    }
  };
  states.maybeprop = function(type2, stream, state) {
    if (type2 == ":")
      return pushContext(state, stream, "prop");
    return pass(type2, stream, state);
  };
  states.prop = function(type2, stream, state) {
    if (type2 == ";")
      return popContext(state);
    if (type2 == "{" && allowNested)
      return pushContext(state, stream, "propBlock");
    if (type2 == "}" || type2 == "{")
      return popAndPass(type2, stream, state);
    if (type2 == "(")
      return pushContext(state, stream, "parens");
    if (type2 == "hash" && !/^#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/.test(stream.current())) {
      override = "error";
    } else if (type2 == "word") {
      wordAsValue(stream);
    } else if (type2 == "interpolation") {
      return pushContext(state, stream, "interpolation");
    }
    return "prop";
  };
  states.propBlock = function(type2, _stream, state) {
    if (type2 == "}")
      return popContext(state);
    if (type2 == "word") {
      override = "property";
      return "maybeprop";
    }
    return state.context.type;
  };
  states.parens = function(type2, stream, state) {
    if (type2 == "{" || type2 == "}")
      return popAndPass(type2, stream, state);
    if (type2 == ")")
      return popContext(state);
    if (type2 == "(")
      return pushContext(state, stream, "parens");
    if (type2 == "interpolation")
      return pushContext(state, stream, "interpolation");
    if (type2 == "word")
      wordAsValue(stream);
    return "parens";
  };
  states.pseudo = function(type2, stream, state) {
    if (type2 == "meta")
      return "pseudo";
    if (type2 == "word") {
      override = "variableName.constant";
      return state.context.type;
    }
    return pass(type2, stream, state);
  };
  states.documentTypes = function(type2, stream, state) {
    if (type2 == "word" && documentTypes2.hasOwnProperty(stream.current())) {
      override = "tag";
      return state.context.type;
    } else {
      return states.atBlock(type2, stream, state);
    }
  };
  states.atBlock = function(type2, stream, state) {
    if (type2 == "(")
      return pushContext(state, stream, "atBlock_parens");
    if (type2 == "}" || type2 == ";")
      return popAndPass(type2, stream, state);
    if (type2 == "{")
      return popContext(state) && pushContext(state, stream, allowNested ? "block" : "top");
    if (type2 == "interpolation")
      return pushContext(state, stream, "interpolation");
    if (type2 == "word") {
      var word = stream.current().toLowerCase();
      if (word == "only" || word == "not" || word == "and" || word == "or")
        override = "keyword";
      else if (mediaTypes2.hasOwnProperty(word))
        override = "attribute";
      else if (mediaFeatures2.hasOwnProperty(word))
        override = "property";
      else if (mediaValueKeywords2.hasOwnProperty(word))
        override = "keyword";
      else if (propertyKeywords2.hasOwnProperty(word))
        override = "property";
      else if (nonStandardPropertyKeywords2.hasOwnProperty(word))
        override = highlightNonStandardPropertyKeywords ? "string.special" : "property";
      else if (valueKeywords2.hasOwnProperty(word))
        override = "atom";
      else if (colorKeywords2.hasOwnProperty(word))
        override = "keyword";
      else
        override = "error";
    }
    return state.context.type;
  };
  states.atComponentBlock = function(type2, stream, state) {
    if (type2 == "}")
      return popAndPass(type2, stream, state);
    if (type2 == "{")
      return popContext(state) && pushContext(state, stream, allowNested ? "block" : "top", false);
    if (type2 == "word")
      override = "error";
    return state.context.type;
  };
  states.atBlock_parens = function(type2, stream, state) {
    if (type2 == ")")
      return popContext(state);
    if (type2 == "{" || type2 == "}")
      return popAndPass(type2, stream, state, 2);
    return states.atBlock(type2, stream, state);
  };
  states.restricted_atBlock_before = function(type2, stream, state) {
    if (type2 == "{")
      return pushContext(state, stream, "restricted_atBlock");
    if (type2 == "word" && state.stateArg == "@counter-style") {
      override = "variable";
      return "restricted_atBlock_before";
    }
    return pass(type2, stream, state);
  };
  states.restricted_atBlock = function(type2, stream, state) {
    if (type2 == "}") {
      state.stateArg = null;
      return popContext(state);
    }
    if (type2 == "word") {
      if (state.stateArg == "@font-face" && !fontProperties2.hasOwnProperty(stream.current().toLowerCase()) || state.stateArg == "@counter-style" && !counterDescriptors2.hasOwnProperty(stream.current().toLowerCase()))
        override = "error";
      else
        override = "property";
      return "maybeprop";
    }
    return "restricted_atBlock";
  };
  states.keyframes = function(type2, stream, state) {
    if (type2 == "word") {
      override = "variable";
      return "keyframes";
    }
    if (type2 == "{")
      return pushContext(state, stream, "top");
    return pass(type2, stream, state);
  };
  states.at = function(type2, stream, state) {
    if (type2 == ";")
      return popContext(state);
    if (type2 == "{" || type2 == "}")
      return popAndPass(type2, stream, state);
    if (type2 == "word")
      override = "tag";
    else if (type2 == "hash")
      override = "builtin";
    return "at";
  };
  states.interpolation = function(type2, stream, state) {
    if (type2 == "}")
      return popContext(state);
    if (type2 == "{" || type2 == ";")
      return popAndPass(type2, stream, state);
    if (type2 == "word")
      override = "variable";
    else if (type2 != "variable" && type2 != "(" && type2 != ")")
      override = "error";
    return "interpolation";
  };
  return {
    name: parserConfig.name,
    startState: function() {
      return {
        tokenize: null,
        state: inline ? "block" : "top",
        stateArg: null,
        context: new Context(inline ? "block" : "top", 0, null)
      };
    },
    token: function(stream, state) {
      if (!state.tokenize && stream.eatSpace())
        return null;
      var style = (state.tokenize || tokenBase)(stream, state);
      if (style && typeof style == "object") {
        type = style[1];
        style = style[0];
      }
      override = style;
      if (type != "comment")
        state.state = states[state.state](type, stream, state);
      return override;
    },
    indent: function(state, textAfter, iCx) {
      var cx = state.context, ch = textAfter && textAfter.charAt(0);
      var indent = cx.indent;
      if (cx.type == "prop" && (ch == "}" || ch == ")"))
        cx = cx.prev;
      if (cx.prev) {
        if (ch == "}" && (cx.type == "block" || cx.type == "top" || cx.type == "interpolation" || cx.type == "restricted_atBlock")) {
          cx = cx.prev;
          indent = cx.indent;
        } else if (ch == ")" && (cx.type == "parens" || cx.type == "atBlock_parens") || ch == "{" && (cx.type == "at" || cx.type == "atBlock")) {
          indent = Math.max(0, cx.indent - iCx.unit);
        }
      }
      return indent;
    },
    languageData: {
      indentOnInput: /^\s*\}$/,
      commentTokens: { line: lineComment, block: { open: "/*", close: "*/" } },
      autocomplete: allWords
    }
  };
}
function keySet(array) {
  var keys = {};
  for (var i = 0; i < array.length; ++i) {
    keys[array[i].toLowerCase()] = true;
  }
  return keys;
}
function tokenCComment(stream, state) {
  var maybeEnd = false, ch;
  while ((ch = stream.next()) != null) {
    if (maybeEnd && ch == "/") {
      state.tokenize = null;
      break;
    }
    maybeEnd = ch == "*";
  }
  return ["comment", "comment"];
}
var documentTypes_, documentTypes, mediaTypes_, mediaTypes, mediaFeatures_, mediaFeatures, mediaValueKeywords_, mediaValueKeywords, propertyKeywords_, propertyKeywords, nonStandardPropertyKeywords_, nonStandardPropertyKeywords, fontProperties_, fontProperties, counterDescriptors_, counterDescriptors, colorKeywords_, colorKeywords, valueKeywords_, valueKeywords, allWords, keywords, defaults, css, sCSS, less, gss;
var init_css = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/css.js"() {
    documentTypes_ = [
      "domain",
      "regexp",
      "url",
      "url-prefix"
    ];
    documentTypes = keySet(documentTypes_);
    mediaTypes_ = [
      "all",
      "aural",
      "braille",
      "handheld",
      "print",
      "projection",
      "screen",
      "tty",
      "tv",
      "embossed"
    ];
    mediaTypes = keySet(mediaTypes_);
    mediaFeatures_ = [
      "width",
      "min-width",
      "max-width",
      "height",
      "min-height",
      "max-height",
      "device-width",
      "min-device-width",
      "max-device-width",
      "device-height",
      "min-device-height",
      "max-device-height",
      "aspect-ratio",
      "min-aspect-ratio",
      "max-aspect-ratio",
      "device-aspect-ratio",
      "min-device-aspect-ratio",
      "max-device-aspect-ratio",
      "color",
      "min-color",
      "max-color",
      "color-index",
      "min-color-index",
      "max-color-index",
      "monochrome",
      "min-monochrome",
      "max-monochrome",
      "resolution",
      "min-resolution",
      "max-resolution",
      "scan",
      "grid",
      "orientation",
      "device-pixel-ratio",
      "min-device-pixel-ratio",
      "max-device-pixel-ratio",
      "pointer",
      "any-pointer",
      "hover",
      "any-hover",
      "prefers-color-scheme",
      "dynamic-range",
      "video-dynamic-range"
    ];
    mediaFeatures = keySet(mediaFeatures_);
    mediaValueKeywords_ = [
      "landscape",
      "portrait",
      "none",
      "coarse",
      "fine",
      "on-demand",
      "hover",
      "interlace",
      "progressive",
      "dark",
      "light",
      "standard",
      "high"
    ];
    mediaValueKeywords = keySet(mediaValueKeywords_);
    propertyKeywords_ = [
      "align-content",
      "align-items",
      "align-self",
      "alignment-adjust",
      "alignment-baseline",
      "all",
      "anchor-point",
      "animation",
      "animation-delay",
      "animation-direction",
      "animation-duration",
      "animation-fill-mode",
      "animation-iteration-count",
      "animation-name",
      "animation-play-state",
      "animation-timing-function",
      "appearance",
      "azimuth",
      "backdrop-filter",
      "backface-visibility",
      "background",
      "background-attachment",
      "background-blend-mode",
      "background-clip",
      "background-color",
      "background-image",
      "background-origin",
      "background-position",
      "background-position-x",
      "background-position-y",
      "background-repeat",
      "background-size",
      "baseline-shift",
      "binding",
      "bleed",
      "block-size",
      "bookmark-label",
      "bookmark-level",
      "bookmark-state",
      "bookmark-target",
      "border",
      "border-bottom",
      "border-bottom-color",
      "border-bottom-left-radius",
      "border-bottom-right-radius",
      "border-bottom-style",
      "border-bottom-width",
      "border-collapse",
      "border-color",
      "border-image",
      "border-image-outset",
      "border-image-repeat",
      "border-image-slice",
      "border-image-source",
      "border-image-width",
      "border-left",
      "border-left-color",
      "border-left-style",
      "border-left-width",
      "border-radius",
      "border-right",
      "border-right-color",
      "border-right-style",
      "border-right-width",
      "border-spacing",
      "border-style",
      "border-top",
      "border-top-color",
      "border-top-left-radius",
      "border-top-right-radius",
      "border-top-style",
      "border-top-width",
      "border-width",
      "bottom",
      "box-decoration-break",
      "box-shadow",
      "box-sizing",
      "break-after",
      "break-before",
      "break-inside",
      "caption-side",
      "caret-color",
      "clear",
      "clip",
      "color",
      "color-profile",
      "column-count",
      "column-fill",
      "column-gap",
      "column-rule",
      "column-rule-color",
      "column-rule-style",
      "column-rule-width",
      "column-span",
      "column-width",
      "columns",
      "contain",
      "content",
      "counter-increment",
      "counter-reset",
      "crop",
      "cue",
      "cue-after",
      "cue-before",
      "cursor",
      "direction",
      "display",
      "dominant-baseline",
      "drop-initial-after-adjust",
      "drop-initial-after-align",
      "drop-initial-before-adjust",
      "drop-initial-before-align",
      "drop-initial-size",
      "drop-initial-value",
      "elevation",
      "empty-cells",
      "fit",
      "fit-content",
      "fit-position",
      "flex",
      "flex-basis",
      "flex-direction",
      "flex-flow",
      "flex-grow",
      "flex-shrink",
      "flex-wrap",
      "float",
      "float-offset",
      "flow-from",
      "flow-into",
      "font",
      "font-family",
      "font-feature-settings",
      "font-kerning",
      "font-language-override",
      "font-optical-sizing",
      "font-size",
      "font-size-adjust",
      "font-stretch",
      "font-style",
      "font-synthesis",
      "font-variant",
      "font-variant-alternates",
      "font-variant-caps",
      "font-variant-east-asian",
      "font-variant-ligatures",
      "font-variant-numeric",
      "font-variant-position",
      "font-variation-settings",
      "font-weight",
      "gap",
      "grid",
      "grid-area",
      "grid-auto-columns",
      "grid-auto-flow",
      "grid-auto-rows",
      "grid-column",
      "grid-column-end",
      "grid-column-gap",
      "grid-column-start",
      "grid-gap",
      "grid-row",
      "grid-row-end",
      "grid-row-gap",
      "grid-row-start",
      "grid-template",
      "grid-template-areas",
      "grid-template-columns",
      "grid-template-rows",
      "hanging-punctuation",
      "height",
      "hyphens",
      "icon",
      "image-orientation",
      "image-rendering",
      "image-resolution",
      "inline-box-align",
      "inset",
      "inset-block",
      "inset-block-end",
      "inset-block-start",
      "inset-inline",
      "inset-inline-end",
      "inset-inline-start",
      "isolation",
      "justify-content",
      "justify-items",
      "justify-self",
      "left",
      "letter-spacing",
      "line-break",
      "line-height",
      "line-height-step",
      "line-stacking",
      "line-stacking-ruby",
      "line-stacking-shift",
      "line-stacking-strategy",
      "list-style",
      "list-style-image",
      "list-style-position",
      "list-style-type",
      "margin",
      "margin-bottom",
      "margin-left",
      "margin-right",
      "margin-top",
      "marks",
      "marquee-direction",
      "marquee-loop",
      "marquee-play-count",
      "marquee-speed",
      "marquee-style",
      "mask-clip",
      "mask-composite",
      "mask-image",
      "mask-mode",
      "mask-origin",
      "mask-position",
      "mask-repeat",
      "mask-size",
      "mask-type",
      "max-block-size",
      "max-height",
      "max-inline-size",
      "max-width",
      "min-block-size",
      "min-height",
      "min-inline-size",
      "min-width",
      "mix-blend-mode",
      "move-to",
      "nav-down",
      "nav-index",
      "nav-left",
      "nav-right",
      "nav-up",
      "object-fit",
      "object-position",
      "offset",
      "offset-anchor",
      "offset-distance",
      "offset-path",
      "offset-position",
      "offset-rotate",
      "opacity",
      "order",
      "orphans",
      "outline",
      "outline-color",
      "outline-offset",
      "outline-style",
      "outline-width",
      "overflow",
      "overflow-style",
      "overflow-wrap",
      "overflow-x",
      "overflow-y",
      "padding",
      "padding-bottom",
      "padding-left",
      "padding-right",
      "padding-top",
      "page",
      "page-break-after",
      "page-break-before",
      "page-break-inside",
      "page-policy",
      "pause",
      "pause-after",
      "pause-before",
      "perspective",
      "perspective-origin",
      "pitch",
      "pitch-range",
      "place-content",
      "place-items",
      "place-self",
      "play-during",
      "position",
      "presentation-level",
      "punctuation-trim",
      "quotes",
      "region-break-after",
      "region-break-before",
      "region-break-inside",
      "region-fragment",
      "rendering-intent",
      "resize",
      "rest",
      "rest-after",
      "rest-before",
      "richness",
      "right",
      "rotate",
      "rotation",
      "rotation-point",
      "row-gap",
      "ruby-align",
      "ruby-overhang",
      "ruby-position",
      "ruby-span",
      "scale",
      "scroll-behavior",
      "scroll-margin",
      "scroll-margin-block",
      "scroll-margin-block-end",
      "scroll-margin-block-start",
      "scroll-margin-bottom",
      "scroll-margin-inline",
      "scroll-margin-inline-end",
      "scroll-margin-inline-start",
      "scroll-margin-left",
      "scroll-margin-right",
      "scroll-margin-top",
      "scroll-padding",
      "scroll-padding-block",
      "scroll-padding-block-end",
      "scroll-padding-block-start",
      "scroll-padding-bottom",
      "scroll-padding-inline",
      "scroll-padding-inline-end",
      "scroll-padding-inline-start",
      "scroll-padding-left",
      "scroll-padding-right",
      "scroll-padding-top",
      "scroll-snap-align",
      "scroll-snap-type",
      "shape-image-threshold",
      "shape-inside",
      "shape-margin",
      "shape-outside",
      "size",
      "speak",
      "speak-as",
      "speak-header",
      "speak-numeral",
      "speak-punctuation",
      "speech-rate",
      "stress",
      "string-set",
      "tab-size",
      "table-layout",
      "target",
      "target-name",
      "target-new",
      "target-position",
      "text-align",
      "text-align-last",
      "text-combine-upright",
      "text-decoration",
      "text-decoration-color",
      "text-decoration-line",
      "text-decoration-skip",
      "text-decoration-skip-ink",
      "text-decoration-style",
      "text-emphasis",
      "text-emphasis-color",
      "text-emphasis-position",
      "text-emphasis-style",
      "text-height",
      "text-indent",
      "text-justify",
      "text-orientation",
      "text-outline",
      "text-overflow",
      "text-rendering",
      "text-shadow",
      "text-size-adjust",
      "text-space-collapse",
      "text-transform",
      "text-underline-position",
      "text-wrap",
      "top",
      "touch-action",
      "transform",
      "transform-origin",
      "transform-style",
      "transition",
      "transition-delay",
      "transition-duration",
      "transition-property",
      "transition-timing-function",
      "translate",
      "unicode-bidi",
      "user-select",
      "vertical-align",
      "visibility",
      "voice-balance",
      "voice-duration",
      "voice-family",
      "voice-pitch",
      "voice-range",
      "voice-rate",
      "voice-stress",
      "voice-volume",
      "volume",
      "white-space",
      "widows",
      "width",
      "will-change",
      "word-break",
      "word-spacing",
      "word-wrap",
      "writing-mode",
      "z-index",
      // SVG-specific
      "clip-path",
      "clip-rule",
      "mask",
      "enable-background",
      "filter",
      "flood-color",
      "flood-opacity",
      "lighting-color",
      "stop-color",
      "stop-opacity",
      "pointer-events",
      "color-interpolation",
      "color-interpolation-filters",
      "color-rendering",
      "fill",
      "fill-opacity",
      "fill-rule",
      "image-rendering",
      "marker",
      "marker-end",
      "marker-mid",
      "marker-start",
      "paint-order",
      "shape-rendering",
      "stroke",
      "stroke-dasharray",
      "stroke-dashoffset",
      "stroke-linecap",
      "stroke-linejoin",
      "stroke-miterlimit",
      "stroke-opacity",
      "stroke-width",
      "text-rendering",
      "baseline-shift",
      "dominant-baseline",
      "glyph-orientation-horizontal",
      "glyph-orientation-vertical",
      "text-anchor",
      "writing-mode"
    ];
    propertyKeywords = keySet(propertyKeywords_);
    nonStandardPropertyKeywords_ = [
      "accent-color",
      "aspect-ratio",
      "border-block",
      "border-block-color",
      "border-block-end",
      "border-block-end-color",
      "border-block-end-style",
      "border-block-end-width",
      "border-block-start",
      "border-block-start-color",
      "border-block-start-style",
      "border-block-start-width",
      "border-block-style",
      "border-block-width",
      "border-inline",
      "border-inline-color",
      "border-inline-end",
      "border-inline-end-color",
      "border-inline-end-style",
      "border-inline-end-width",
      "border-inline-start",
      "border-inline-start-color",
      "border-inline-start-style",
      "border-inline-start-width",
      "border-inline-style",
      "border-inline-width",
      "content-visibility",
      "margin-block",
      "margin-block-end",
      "margin-block-start",
      "margin-inline",
      "margin-inline-end",
      "margin-inline-start",
      "overflow-anchor",
      "overscroll-behavior",
      "padding-block",
      "padding-block-end",
      "padding-block-start",
      "padding-inline",
      "padding-inline-end",
      "padding-inline-start",
      "scroll-snap-stop",
      "scrollbar-3d-light-color",
      "scrollbar-arrow-color",
      "scrollbar-base-color",
      "scrollbar-dark-shadow-color",
      "scrollbar-face-color",
      "scrollbar-highlight-color",
      "scrollbar-shadow-color",
      "scrollbar-track-color",
      "searchfield-cancel-button",
      "searchfield-decoration",
      "searchfield-results-button",
      "searchfield-results-decoration",
      "shape-inside",
      "zoom"
    ];
    nonStandardPropertyKeywords = keySet(nonStandardPropertyKeywords_);
    fontProperties_ = [
      "font-display",
      "font-family",
      "src",
      "unicode-range",
      "font-variant",
      "font-feature-settings",
      "font-stretch",
      "font-weight",
      "font-style"
    ];
    fontProperties = keySet(fontProperties_);
    counterDescriptors_ = [
      "additive-symbols",
      "fallback",
      "negative",
      "pad",
      "prefix",
      "range",
      "speak-as",
      "suffix",
      "symbols",
      "system"
    ];
    counterDescriptors = keySet(counterDescriptors_);
    colorKeywords_ = [
      "aliceblue",
      "antiquewhite",
      "aqua",
      "aquamarine",
      "azure",
      "beige",
      "bisque",
      "black",
      "blanchedalmond",
      "blue",
      "blueviolet",
      "brown",
      "burlywood",
      "cadetblue",
      "chartreuse",
      "chocolate",
      "coral",
      "cornflowerblue",
      "cornsilk",
      "crimson",
      "cyan",
      "darkblue",
      "darkcyan",
      "darkgoldenrod",
      "darkgray",
      "darkgreen",
      "darkgrey",
      "darkkhaki",
      "darkmagenta",
      "darkolivegreen",
      "darkorange",
      "darkorchid",
      "darkred",
      "darksalmon",
      "darkseagreen",
      "darkslateblue",
      "darkslategray",
      "darkslategrey",
      "darkturquoise",
      "darkviolet",
      "deeppink",
      "deepskyblue",
      "dimgray",
      "dimgrey",
      "dodgerblue",
      "firebrick",
      "floralwhite",
      "forestgreen",
      "fuchsia",
      "gainsboro",
      "ghostwhite",
      "gold",
      "goldenrod",
      "gray",
      "grey",
      "green",
      "greenyellow",
      "honeydew",
      "hotpink",
      "indianred",
      "indigo",
      "ivory",
      "khaki",
      "lavender",
      "lavenderblush",
      "lawngreen",
      "lemonchiffon",
      "lightblue",
      "lightcoral",
      "lightcyan",
      "lightgoldenrodyellow",
      "lightgray",
      "lightgreen",
      "lightgrey",
      "lightpink",
      "lightsalmon",
      "lightseagreen",
      "lightskyblue",
      "lightslategray",
      "lightslategrey",
      "lightsteelblue",
      "lightyellow",
      "lime",
      "limegreen",
      "linen",
      "magenta",
      "maroon",
      "mediumaquamarine",
      "mediumblue",
      "mediumorchid",
      "mediumpurple",
      "mediumseagreen",
      "mediumslateblue",
      "mediumspringgreen",
      "mediumturquoise",
      "mediumvioletred",
      "midnightblue",
      "mintcream",
      "mistyrose",
      "moccasin",
      "navajowhite",
      "navy",
      "oldlace",
      "olive",
      "olivedrab",
      "orange",
      "orangered",
      "orchid",
      "palegoldenrod",
      "palegreen",
      "paleturquoise",
      "palevioletred",
      "papayawhip",
      "peachpuff",
      "peru",
      "pink",
      "plum",
      "powderblue",
      "purple",
      "rebeccapurple",
      "red",
      "rosybrown",
      "royalblue",
      "saddlebrown",
      "salmon",
      "sandybrown",
      "seagreen",
      "seashell",
      "sienna",
      "silver",
      "skyblue",
      "slateblue",
      "slategray",
      "slategrey",
      "snow",
      "springgreen",
      "steelblue",
      "tan",
      "teal",
      "thistle",
      "tomato",
      "turquoise",
      "violet",
      "wheat",
      "white",
      "whitesmoke",
      "yellow",
      "yellowgreen"
    ];
    colorKeywords = keySet(colorKeywords_);
    valueKeywords_ = [
      "above",
      "absolute",
      "activeborder",
      "additive",
      "activecaption",
      "afar",
      "after-white-space",
      "ahead",
      "alias",
      "all",
      "all-scroll",
      "alphabetic",
      "alternate",
      "always",
      "amharic",
      "amharic-abegede",
      "antialiased",
      "appworkspace",
      "arabic-indic",
      "armenian",
      "asterisks",
      "attr",
      "auto",
      "auto-flow",
      "avoid",
      "avoid-column",
      "avoid-page",
      "avoid-region",
      "axis-pan",
      "background",
      "backwards",
      "baseline",
      "below",
      "bidi-override",
      "binary",
      "bengali",
      "blink",
      "block",
      "block-axis",
      "blur",
      "bold",
      "bolder",
      "border",
      "border-box",
      "both",
      "bottom",
      "break",
      "break-all",
      "break-word",
      "brightness",
      "bullets",
      "button",
      "buttonface",
      "buttonhighlight",
      "buttonshadow",
      "buttontext",
      "calc",
      "cambodian",
      "capitalize",
      "caps-lock-indicator",
      "caption",
      "captiontext",
      "caret",
      "cell",
      "center",
      "checkbox",
      "circle",
      "cjk-decimal",
      "cjk-earthly-branch",
      "cjk-heavenly-stem",
      "cjk-ideographic",
      "clear",
      "clip",
      "close-quote",
      "col-resize",
      "collapse",
      "color",
      "color-burn",
      "color-dodge",
      "column",
      "column-reverse",
      "compact",
      "condensed",
      "conic-gradient",
      "contain",
      "content",
      "contents",
      "content-box",
      "context-menu",
      "continuous",
      "contrast",
      "copy",
      "counter",
      "counters",
      "cover",
      "crop",
      "cross",
      "crosshair",
      "cubic-bezier",
      "currentcolor",
      "cursive",
      "cyclic",
      "darken",
      "dashed",
      "decimal",
      "decimal-leading-zero",
      "default",
      "default-button",
      "dense",
      "destination-atop",
      "destination-in",
      "destination-out",
      "destination-over",
      "devanagari",
      "difference",
      "disc",
      "discard",
      "disclosure-closed",
      "disclosure-open",
      "document",
      "dot-dash",
      "dot-dot-dash",
      "dotted",
      "double",
      "down",
      "drop-shadow",
      "e-resize",
      "ease",
      "ease-in",
      "ease-in-out",
      "ease-out",
      "element",
      "ellipse",
      "ellipsis",
      "embed",
      "end",
      "ethiopic",
      "ethiopic-abegede",
      "ethiopic-abegede-am-et",
      "ethiopic-abegede-gez",
      "ethiopic-abegede-ti-er",
      "ethiopic-abegede-ti-et",
      "ethiopic-halehame-aa-er",
      "ethiopic-halehame-aa-et",
      "ethiopic-halehame-am-et",
      "ethiopic-halehame-gez",
      "ethiopic-halehame-om-et",
      "ethiopic-halehame-sid-et",
      "ethiopic-halehame-so-et",
      "ethiopic-halehame-ti-er",
      "ethiopic-halehame-ti-et",
      "ethiopic-halehame-tig",
      "ethiopic-numeric",
      "ew-resize",
      "exclusion",
      "expanded",
      "extends",
      "extra-condensed",
      "extra-expanded",
      "fantasy",
      "fast",
      "fill",
      "fill-box",
      "fixed",
      "flat",
      "flex",
      "flex-end",
      "flex-start",
      "footnotes",
      "forwards",
      "from",
      "geometricPrecision",
      "georgian",
      "grayscale",
      "graytext",
      "grid",
      "groove",
      "gujarati",
      "gurmukhi",
      "hand",
      "hangul",
      "hangul-consonant",
      "hard-light",
      "hebrew",
      "help",
      "hidden",
      "hide",
      "higher",
      "highlight",
      "highlighttext",
      "hiragana",
      "hiragana-iroha",
      "horizontal",
      "hsl",
      "hsla",
      "hue",
      "hue-rotate",
      "icon",
      "ignore",
      "inactiveborder",
      "inactivecaption",
      "inactivecaptiontext",
      "infinite",
      "infobackground",
      "infotext",
      "inherit",
      "initial",
      "inline",
      "inline-axis",
      "inline-block",
      "inline-flex",
      "inline-grid",
      "inline-table",
      "inset",
      "inside",
      "intrinsic",
      "invert",
      "italic",
      "japanese-formal",
      "japanese-informal",
      "justify",
      "kannada",
      "katakana",
      "katakana-iroha",
      "keep-all",
      "khmer",
      "korean-hangul-formal",
      "korean-hanja-formal",
      "korean-hanja-informal",
      "landscape",
      "lao",
      "large",
      "larger",
      "left",
      "level",
      "lighter",
      "lighten",
      "line-through",
      "linear",
      "linear-gradient",
      "lines",
      "list-item",
      "listbox",
      "listitem",
      "local",
      "logical",
      "loud",
      "lower",
      "lower-alpha",
      "lower-armenian",
      "lower-greek",
      "lower-hexadecimal",
      "lower-latin",
      "lower-norwegian",
      "lower-roman",
      "lowercase",
      "ltr",
      "luminosity",
      "malayalam",
      "manipulation",
      "match",
      "matrix",
      "matrix3d",
      "media-play-button",
      "media-slider",
      "media-sliderthumb",
      "media-volume-slider",
      "media-volume-sliderthumb",
      "medium",
      "menu",
      "menulist",
      "menulist-button",
      "menutext",
      "message-box",
      "middle",
      "min-intrinsic",
      "mix",
      "mongolian",
      "monospace",
      "move",
      "multiple",
      "multiple_mask_images",
      "multiply",
      "myanmar",
      "n-resize",
      "narrower",
      "ne-resize",
      "nesw-resize",
      "no-close-quote",
      "no-drop",
      "no-open-quote",
      "no-repeat",
      "none",
      "normal",
      "not-allowed",
      "nowrap",
      "ns-resize",
      "numbers",
      "numeric",
      "nw-resize",
      "nwse-resize",
      "oblique",
      "octal",
      "opacity",
      "open-quote",
      "optimizeLegibility",
      "optimizeSpeed",
      "oriya",
      "oromo",
      "outset",
      "outside",
      "outside-shape",
      "overlay",
      "overline",
      "padding",
      "padding-box",
      "painted",
      "page",
      "paused",
      "persian",
      "perspective",
      "pinch-zoom",
      "plus-darker",
      "plus-lighter",
      "pointer",
      "polygon",
      "portrait",
      "pre",
      "pre-line",
      "pre-wrap",
      "preserve-3d",
      "progress",
      "push-button",
      "radial-gradient",
      "radio",
      "read-only",
      "read-write",
      "read-write-plaintext-only",
      "rectangle",
      "region",
      "relative",
      "repeat",
      "repeating-linear-gradient",
      "repeating-radial-gradient",
      "repeating-conic-gradient",
      "repeat-x",
      "repeat-y",
      "reset",
      "reverse",
      "rgb",
      "rgba",
      "ridge",
      "right",
      "rotate",
      "rotate3d",
      "rotateX",
      "rotateY",
      "rotateZ",
      "round",
      "row",
      "row-resize",
      "row-reverse",
      "rtl",
      "run-in",
      "running",
      "s-resize",
      "sans-serif",
      "saturate",
      "saturation",
      "scale",
      "scale3d",
      "scaleX",
      "scaleY",
      "scaleZ",
      "screen",
      "scroll",
      "scrollbar",
      "scroll-position",
      "se-resize",
      "searchfield",
      "searchfield-cancel-button",
      "searchfield-decoration",
      "searchfield-results-button",
      "searchfield-results-decoration",
      "self-start",
      "self-end",
      "semi-condensed",
      "semi-expanded",
      "separate",
      "sepia",
      "serif",
      "show",
      "sidama",
      "simp-chinese-formal",
      "simp-chinese-informal",
      "single",
      "skew",
      "skewX",
      "skewY",
      "skip-white-space",
      "slide",
      "slider-horizontal",
      "slider-vertical",
      "sliderthumb-horizontal",
      "sliderthumb-vertical",
      "slow",
      "small",
      "small-caps",
      "small-caption",
      "smaller",
      "soft-light",
      "solid",
      "somali",
      "source-atop",
      "source-in",
      "source-out",
      "source-over",
      "space",
      "space-around",
      "space-between",
      "space-evenly",
      "spell-out",
      "square",
      "square-button",
      "start",
      "static",
      "status-bar",
      "stretch",
      "stroke",
      "stroke-box",
      "sub",
      "subpixel-antialiased",
      "svg_masks",
      "super",
      "sw-resize",
      "symbolic",
      "symbols",
      "system-ui",
      "table",
      "table-caption",
      "table-cell",
      "table-column",
      "table-column-group",
      "table-footer-group",
      "table-header-group",
      "table-row",
      "table-row-group",
      "tamil",
      "telugu",
      "text",
      "text-bottom",
      "text-top",
      "textarea",
      "textfield",
      "thai",
      "thick",
      "thin",
      "threeddarkshadow",
      "threedface",
      "threedhighlight",
      "threedlightshadow",
      "threedshadow",
      "tibetan",
      "tigre",
      "tigrinya-er",
      "tigrinya-er-abegede",
      "tigrinya-et",
      "tigrinya-et-abegede",
      "to",
      "top",
      "trad-chinese-formal",
      "trad-chinese-informal",
      "transform",
      "translate",
      "translate3d",
      "translateX",
      "translateY",
      "translateZ",
      "transparent",
      "ultra-condensed",
      "ultra-expanded",
      "underline",
      "unidirectional-pan",
      "unset",
      "up",
      "upper-alpha",
      "upper-armenian",
      "upper-greek",
      "upper-hexadecimal",
      "upper-latin",
      "upper-norwegian",
      "upper-roman",
      "uppercase",
      "urdu",
      "url",
      "var",
      "vertical",
      "vertical-text",
      "view-box",
      "visible",
      "visibleFill",
      "visiblePainted",
      "visibleStroke",
      "visual",
      "w-resize",
      "wait",
      "wave",
      "wider",
      "window",
      "windowframe",
      "windowtext",
      "words",
      "wrap",
      "wrap-reverse",
      "x-large",
      "x-small",
      "xor",
      "xx-large",
      "xx-small"
    ];
    valueKeywords = keySet(valueKeywords_);
    allWords = documentTypes_.concat(mediaTypes_).concat(mediaFeatures_).concat(mediaValueKeywords_).concat(propertyKeywords_).concat(nonStandardPropertyKeywords_).concat(colorKeywords_).concat(valueKeywords_);
    keywords = {
      properties: propertyKeywords_,
      colors: colorKeywords_,
      fonts: fontProperties_,
      values: valueKeywords_,
      all: allWords
    };
    defaults = {
      documentTypes,
      mediaTypes,
      mediaFeatures,
      mediaValueKeywords,
      propertyKeywords,
      nonStandardPropertyKeywords,
      fontProperties,
      counterDescriptors,
      colorKeywords,
      valueKeywords,
      tokenHooks: {
        "/": function(stream, state) {
          if (!stream.eat("*"))
            return false;
          state.tokenize = tokenCComment;
          return tokenCComment(stream, state);
        }
      }
    };
    css = mkCSS({ name: "css" });
    sCSS = mkCSS({
      name: "scss",
      mediaTypes,
      mediaFeatures,
      mediaValueKeywords,
      propertyKeywords,
      nonStandardPropertyKeywords,
      colorKeywords,
      valueKeywords,
      fontProperties,
      allowNested: true,
      lineComment: "//",
      tokenHooks: {
        "/": function(stream, state) {
          if (stream.eat("/")) {
            stream.skipToEnd();
            return ["comment", "comment"];
          } else if (stream.eat("*")) {
            state.tokenize = tokenCComment;
            return tokenCComment(stream, state);
          } else {
            return ["operator", "operator"];
          }
        },
        ":": function(stream) {
          if (stream.match(/^\s*\{/, false))
            return [null, null];
          return false;
        },
        "$": function(stream) {
          stream.match(/^[\w-]+/);
          if (stream.match(/^\s*:/, false))
            return ["def", "variable-definition"];
          return ["variableName.special", "variable"];
        },
        "#": function(stream) {
          if (!stream.eat("{"))
            return false;
          return [null, "interpolation"];
        }
      }
    });
    less = mkCSS({
      name: "less",
      mediaTypes,
      mediaFeatures,
      mediaValueKeywords,
      propertyKeywords,
      nonStandardPropertyKeywords,
      colorKeywords,
      valueKeywords,
      fontProperties,
      allowNested: true,
      lineComment: "//",
      tokenHooks: {
        "/": function(stream, state) {
          if (stream.eat("/")) {
            stream.skipToEnd();
            return ["comment", "comment"];
          } else if (stream.eat("*")) {
            state.tokenize = tokenCComment;
            return tokenCComment(stream, state);
          } else {
            return ["operator", "operator"];
          }
        },
        "@": function(stream) {
          if (stream.eat("{"))
            return [null, "interpolation"];
          if (stream.match(/^(charset|document|font-face|import|(-(moz|ms|o|webkit)-)?keyframes|media|namespace|page|supports)\b/i, false))
            return false;
          stream.eatWhile(/[\w\\\-]/);
          if (stream.match(/^\s*:/, false))
            return ["def", "variable-definition"];
          return ["variableName", "variable"];
        },
        "&": function() {
          return ["atom", "atom"];
        }
      }
    });
    gss = mkCSS({
      name: "gss",
      documentTypes,
      mediaTypes,
      mediaFeatures,
      propertyKeywords,
      nonStandardPropertyKeywords,
      fontProperties,
      counterDescriptors,
      colorKeywords,
      valueKeywords,
      supportsAtComponent: true,
      tokenHooks: {
        "/": function(stream, state) {
          if (!stream.eat("*"))
            return false;
          state.tokenize = tokenCComment;
          return tokenCComment(stream, state);
        }
      }
    });
  }
});
init_css();
export {
  css,
  gss,
  keywords,
  less,
  mkCSS,
  sCSS
};
//# sourceMappingURL=/build/_shared/css-FYV36AOE.js.map
