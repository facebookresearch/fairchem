import {
  EXIT,
  SKIP,
  convert,
  visit
} from "/build/_shared/chunk-LPSXN4QV.js";
import {
  __commonJS,
  __toESM
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/boolbase/index.js
var require_boolbase = __commonJS({
  "../../node_modules/boolbase/index.js"(exports, module) {
    module.exports = {
      trueFunc: function trueFunc() {
        return true;
      },
      falseFunc: function falseFunc() {
        return false;
      }
    };
  }
});

// ../../node_modules/css-selector-parser/lib/utils.js
var require_utils = __commonJS({
  "../../node_modules/css-selector-parser/lib/utils.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    function isIdentStart(c) {
      return c >= "a" && c <= "z" || c >= "A" && c <= "Z" || c === "-" || c === "_";
    }
    exports.isIdentStart = isIdentStart;
    function isIdent(c) {
      return c >= "a" && c <= "z" || c >= "A" && c <= "Z" || c >= "0" && c <= "9" || c === "-" || c === "_";
    }
    exports.isIdent = isIdent;
    function isHex(c) {
      return c >= "a" && c <= "f" || c >= "A" && c <= "F" || c >= "0" && c <= "9";
    }
    exports.isHex = isHex;
    function escapeIdentifier(s) {
      var len = s.length;
      var result = "";
      var i = 0;
      while (i < len) {
        var chr = s.charAt(i);
        if (exports.identSpecialChars[chr]) {
          result += "\\" + chr;
        } else {
          if (!(chr === "_" || chr === "-" || chr >= "A" && chr <= "Z" || chr >= "a" && chr <= "z" || i !== 0 && chr >= "0" && chr <= "9")) {
            var charCode = chr.charCodeAt(0);
            if ((charCode & 63488) === 55296) {
              var extraCharCode = s.charCodeAt(i++);
              if ((charCode & 64512) !== 55296 || (extraCharCode & 64512) !== 56320) {
                throw Error("UCS-2(decode): illegal sequence");
              }
              charCode = ((charCode & 1023) << 10) + (extraCharCode & 1023) + 65536;
            }
            result += "\\" + charCode.toString(16) + " ";
          } else {
            result += chr;
          }
        }
        i++;
      }
      return result;
    }
    exports.escapeIdentifier = escapeIdentifier;
    function escapeStr(s) {
      var len = s.length;
      var result = "";
      var i = 0;
      var replacement;
      while (i < len) {
        var chr = s.charAt(i);
        if (chr === '"') {
          chr = '\\"';
        } else if (chr === "\\") {
          chr = "\\\\";
        } else if ((replacement = exports.strReplacementsRev[chr]) !== void 0) {
          chr = replacement;
        }
        result += chr;
        i++;
      }
      return '"' + result + '"';
    }
    exports.escapeStr = escapeStr;
    exports.identSpecialChars = {
      "!": true,
      '"': true,
      "#": true,
      "$": true,
      "%": true,
      "&": true,
      "'": true,
      "(": true,
      ")": true,
      "*": true,
      "+": true,
      ",": true,
      ".": true,
      "/": true,
      ";": true,
      "<": true,
      "=": true,
      ">": true,
      "?": true,
      "@": true,
      "[": true,
      "\\": true,
      "]": true,
      "^": true,
      "`": true,
      "{": true,
      "|": true,
      "}": true,
      "~": true
    };
    exports.strReplacementsRev = {
      "\n": "\\n",
      "\r": "\\r",
      "	": "\\t",
      "\f": "\\f",
      "\v": "\\v"
    };
    exports.singleQuoteEscapeChars = {
      n: "\n",
      r: "\r",
      t: "	",
      f: "\f",
      "\\": "\\",
      "'": "'"
    };
    exports.doubleQuotesEscapeChars = {
      n: "\n",
      r: "\r",
      t: "	",
      f: "\f",
      "\\": "\\",
      '"': '"'
    };
  }
});

// ../../node_modules/css-selector-parser/lib/parser-context.js
var require_parser_context = __commonJS({
  "../../node_modules/css-selector-parser/lib/parser-context.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var utils_1 = require_utils();
    function parseCssSelector(str, pos, pseudos, attrEqualityMods, ruleNestingOperators, substitutesEnabled) {
      var l = str.length;
      var chr = "";
      function getStr(quote, escapeTable) {
        var result = "";
        pos++;
        chr = str.charAt(pos);
        while (pos < l) {
          if (chr === quote) {
            pos++;
            return result;
          } else if (chr === "\\") {
            pos++;
            chr = str.charAt(pos);
            var esc = void 0;
            if (chr === quote) {
              result += quote;
            } else if ((esc = escapeTable[chr]) !== void 0) {
              result += esc;
            } else if (utils_1.isHex(chr)) {
              var hex = chr;
              pos++;
              chr = str.charAt(pos);
              while (utils_1.isHex(chr)) {
                hex += chr;
                pos++;
                chr = str.charAt(pos);
              }
              if (chr === " ") {
                pos++;
                chr = str.charAt(pos);
              }
              result += String.fromCharCode(parseInt(hex, 16));
              continue;
            } else {
              result += chr;
            }
          } else {
            result += chr;
          }
          pos++;
          chr = str.charAt(pos);
        }
        return result;
      }
      function getIdent() {
        var result = "";
        chr = str.charAt(pos);
        while (pos < l) {
          if (utils_1.isIdent(chr)) {
            result += chr;
          } else if (chr === "\\") {
            pos++;
            if (pos >= l) {
              throw Error("Expected symbol but end of file reached.");
            }
            chr = str.charAt(pos);
            if (utils_1.identSpecialChars[chr]) {
              result += chr;
            } else if (utils_1.isHex(chr)) {
              var hex = chr;
              pos++;
              chr = str.charAt(pos);
              while (utils_1.isHex(chr)) {
                hex += chr;
                pos++;
                chr = str.charAt(pos);
              }
              if (chr === " ") {
                pos++;
                chr = str.charAt(pos);
              }
              result += String.fromCharCode(parseInt(hex, 16));
              continue;
            } else {
              result += chr;
            }
          } else {
            return result;
          }
          pos++;
          chr = str.charAt(pos);
        }
        return result;
      }
      function skipWhitespace() {
        chr = str.charAt(pos);
        var result = false;
        while (chr === " " || chr === "	" || chr === "\n" || chr === "\r" || chr === "\f") {
          result = true;
          pos++;
          chr = str.charAt(pos);
        }
        return result;
      }
      function parse3() {
        var res = parseSelector();
        if (pos < l) {
          throw Error('Rule expected but "' + str.charAt(pos) + '" found.');
        }
        return res;
      }
      function parseSelector() {
        var selector = parseSingleSelector();
        if (!selector) {
          return null;
        }
        var res = selector;
        chr = str.charAt(pos);
        while (chr === ",") {
          pos++;
          skipWhitespace();
          if (res.type !== "selectors") {
            res = {
              type: "selectors",
              selectors: [selector]
            };
          }
          selector = parseSingleSelector();
          if (!selector) {
            throw Error('Rule expected after ",".');
          }
          res.selectors.push(selector);
        }
        return res;
      }
      function parseSingleSelector() {
        skipWhitespace();
        var selector = {
          type: "ruleSet"
        };
        var rule = parseRule();
        if (!rule) {
          return null;
        }
        var currentRule = selector;
        while (rule) {
          rule.type = "rule";
          currentRule.rule = rule;
          currentRule = rule;
          skipWhitespace();
          chr = str.charAt(pos);
          if (pos >= l || chr === "," || chr === ")") {
            break;
          }
          if (ruleNestingOperators[chr]) {
            var op = chr;
            pos++;
            skipWhitespace();
            rule = parseRule();
            if (!rule) {
              throw Error('Rule expected after "' + op + '".');
            }
            rule.nestingOperator = op;
          } else {
            rule = parseRule();
            if (rule) {
              rule.nestingOperator = null;
            }
          }
        }
        return selector;
      }
      function parseRule() {
        var rule = null;
        while (pos < l) {
          chr = str.charAt(pos);
          if (chr === "*") {
            pos++;
            (rule = rule || {}).tagName = "*";
          } else if (utils_1.isIdentStart(chr) || chr === "\\") {
            (rule = rule || {}).tagName = getIdent();
          } else if (chr === ".") {
            pos++;
            rule = rule || {};
            (rule.classNames = rule.classNames || []).push(getIdent());
          } else if (chr === "#") {
            pos++;
            (rule = rule || {}).id = getIdent();
          } else if (chr === "[") {
            pos++;
            skipWhitespace();
            var attr = {
              name: getIdent()
            };
            skipWhitespace();
            if (chr === "]") {
              pos++;
            } else {
              var operator = "";
              if (attrEqualityMods[chr]) {
                operator = chr;
                pos++;
                chr = str.charAt(pos);
              }
              if (pos >= l) {
                throw Error('Expected "=" but end of file reached.');
              }
              if (chr !== "=") {
                throw Error('Expected "=" but "' + chr + '" found.');
              }
              attr.operator = operator + "=";
              pos++;
              skipWhitespace();
              var attrValue = "";
              attr.valueType = "string";
              if (chr === '"') {
                attrValue = getStr('"', utils_1.doubleQuotesEscapeChars);
              } else if (chr === "'") {
                attrValue = getStr("'", utils_1.singleQuoteEscapeChars);
              } else if (substitutesEnabled && chr === "$") {
                pos++;
                attrValue = getIdent();
                attr.valueType = "substitute";
              } else {
                while (pos < l) {
                  if (chr === "]") {
                    break;
                  }
                  attrValue += chr;
                  pos++;
                  chr = str.charAt(pos);
                }
                attrValue = attrValue.trim();
              }
              skipWhitespace();
              if (pos >= l) {
                throw Error('Expected "]" but end of file reached.');
              }
              if (chr !== "]") {
                throw Error('Expected "]" but "' + chr + '" found.');
              }
              pos++;
              attr.value = attrValue;
            }
            rule = rule || {};
            (rule.attrs = rule.attrs || []).push(attr);
          } else if (chr === ":") {
            pos++;
            var pseudoName = getIdent();
            var pseudo2 = {
              name: pseudoName
            };
            if (chr === "(") {
              pos++;
              var value = "";
              skipWhitespace();
              if (pseudos[pseudoName] === "selector") {
                pseudo2.valueType = "selector";
                value = parseSelector();
              } else {
                pseudo2.valueType = pseudos[pseudoName] || "string";
                if (chr === '"') {
                  value = getStr('"', utils_1.doubleQuotesEscapeChars);
                } else if (chr === "'") {
                  value = getStr("'", utils_1.singleQuoteEscapeChars);
                } else if (substitutesEnabled && chr === "$") {
                  pos++;
                  value = getIdent();
                  pseudo2.valueType = "substitute";
                } else {
                  while (pos < l) {
                    if (chr === ")") {
                      break;
                    }
                    value += chr;
                    pos++;
                    chr = str.charAt(pos);
                  }
                  value = value.trim();
                }
                skipWhitespace();
              }
              if (pos >= l) {
                throw Error('Expected ")" but end of file reached.');
              }
              if (chr !== ")") {
                throw Error('Expected ")" but "' + chr + '" found.');
              }
              pos++;
              pseudo2.value = value;
            }
            rule = rule || {};
            (rule.pseudos = rule.pseudos || []).push(pseudo2);
          } else {
            break;
          }
        }
        return rule;
      }
      return parse3();
    }
    exports.parseCssSelector = parseCssSelector;
  }
});

// ../../node_modules/css-selector-parser/lib/render.js
var require_render = __commonJS({
  "../../node_modules/css-selector-parser/lib/render.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var utils_1 = require_utils();
    function renderEntity(entity) {
      var res = "";
      switch (entity.type) {
        case "ruleSet":
          var currentEntity = entity.rule;
          var parts = [];
          while (currentEntity) {
            if (currentEntity.nestingOperator) {
              parts.push(currentEntity.nestingOperator);
            }
            parts.push(renderEntity(currentEntity));
            currentEntity = currentEntity.rule;
          }
          res = parts.join(" ");
          break;
        case "selectors":
          res = entity.selectors.map(renderEntity).join(", ");
          break;
        case "rule":
          if (entity.tagName) {
            if (entity.tagName === "*") {
              res = "*";
            } else {
              res = utils_1.escapeIdentifier(entity.tagName);
            }
          }
          if (entity.id) {
            res += "#" + utils_1.escapeIdentifier(entity.id);
          }
          if (entity.classNames) {
            res += entity.classNames.map(function(cn) {
              return "." + utils_1.escapeIdentifier(cn);
            }).join("");
          }
          if (entity.attrs) {
            res += entity.attrs.map(function(attr) {
              if ("operator" in attr) {
                if (attr.valueType === "substitute") {
                  return "[" + utils_1.escapeIdentifier(attr.name) + attr.operator + "$" + attr.value + "]";
                } else {
                  return "[" + utils_1.escapeIdentifier(attr.name) + attr.operator + utils_1.escapeStr(attr.value) + "]";
                }
              } else {
                return "[" + utils_1.escapeIdentifier(attr.name) + "]";
              }
            }).join("");
          }
          if (entity.pseudos) {
            res += entity.pseudos.map(function(pseudo2) {
              if (pseudo2.valueType) {
                if (pseudo2.valueType === "selector") {
                  return ":" + utils_1.escapeIdentifier(pseudo2.name) + "(" + renderEntity(pseudo2.value) + ")";
                } else if (pseudo2.valueType === "substitute") {
                  return ":" + utils_1.escapeIdentifier(pseudo2.name) + "($" + pseudo2.value + ")";
                } else if (pseudo2.valueType === "numeric") {
                  return ":" + utils_1.escapeIdentifier(pseudo2.name) + "(" + pseudo2.value + ")";
                } else {
                  return ":" + utils_1.escapeIdentifier(pseudo2.name) + "(" + utils_1.escapeIdentifier(pseudo2.value) + ")";
                }
              } else {
                return ":" + utils_1.escapeIdentifier(pseudo2.name);
              }
            }).join("");
          }
          break;
        default:
          throw Error('Unknown entity type: "' + entity.type + '".');
      }
      return res;
    }
    exports.renderEntity = renderEntity;
  }
});

// ../../node_modules/css-selector-parser/lib/index.js
var require_lib = __commonJS({
  "../../node_modules/css-selector-parser/lib/index.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var parser_context_1 = require_parser_context();
    var render_1 = require_render();
    var CssSelectorParser2 = (
      /** @class */
      function() {
        function CssSelectorParser3() {
          this.pseudos = {};
          this.attrEqualityMods = {};
          this.ruleNestingOperators = {};
          this.substitutesEnabled = false;
        }
        CssSelectorParser3.prototype.registerSelectorPseudos = function() {
          var pseudos = [];
          for (var _i = 0; _i < arguments.length; _i++) {
            pseudos[_i] = arguments[_i];
          }
          for (var _a = 0, pseudos_1 = pseudos; _a < pseudos_1.length; _a++) {
            var pseudo2 = pseudos_1[_a];
            this.pseudos[pseudo2] = "selector";
          }
          return this;
        };
        CssSelectorParser3.prototype.unregisterSelectorPseudos = function() {
          var pseudos = [];
          for (var _i = 0; _i < arguments.length; _i++) {
            pseudos[_i] = arguments[_i];
          }
          for (var _a = 0, pseudos_2 = pseudos; _a < pseudos_2.length; _a++) {
            var pseudo2 = pseudos_2[_a];
            delete this.pseudos[pseudo2];
          }
          return this;
        };
        CssSelectorParser3.prototype.registerNumericPseudos = function() {
          var pseudos = [];
          for (var _i = 0; _i < arguments.length; _i++) {
            pseudos[_i] = arguments[_i];
          }
          for (var _a = 0, pseudos_3 = pseudos; _a < pseudos_3.length; _a++) {
            var pseudo2 = pseudos_3[_a];
            this.pseudos[pseudo2] = "numeric";
          }
          return this;
        };
        CssSelectorParser3.prototype.unregisterNumericPseudos = function() {
          var pseudos = [];
          for (var _i = 0; _i < arguments.length; _i++) {
            pseudos[_i] = arguments[_i];
          }
          for (var _a = 0, pseudos_4 = pseudos; _a < pseudos_4.length; _a++) {
            var pseudo2 = pseudos_4[_a];
            delete this.pseudos[pseudo2];
          }
          return this;
        };
        CssSelectorParser3.prototype.registerNestingOperators = function() {
          var operators = [];
          for (var _i = 0; _i < arguments.length; _i++) {
            operators[_i] = arguments[_i];
          }
          for (var _a = 0, operators_1 = operators; _a < operators_1.length; _a++) {
            var operator = operators_1[_a];
            this.ruleNestingOperators[operator] = true;
          }
          return this;
        };
        CssSelectorParser3.prototype.unregisterNestingOperators = function() {
          var operators = [];
          for (var _i = 0; _i < arguments.length; _i++) {
            operators[_i] = arguments[_i];
          }
          for (var _a = 0, operators_2 = operators; _a < operators_2.length; _a++) {
            var operator = operators_2[_a];
            delete this.ruleNestingOperators[operator];
          }
          return this;
        };
        CssSelectorParser3.prototype.registerAttrEqualityMods = function() {
          var mods = [];
          for (var _i = 0; _i < arguments.length; _i++) {
            mods[_i] = arguments[_i];
          }
          for (var _a = 0, mods_1 = mods; _a < mods_1.length; _a++) {
            var mod = mods_1[_a];
            this.attrEqualityMods[mod] = true;
          }
          return this;
        };
        CssSelectorParser3.prototype.unregisterAttrEqualityMods = function() {
          var mods = [];
          for (var _i = 0; _i < arguments.length; _i++) {
            mods[_i] = arguments[_i];
          }
          for (var _a = 0, mods_2 = mods; _a < mods_2.length; _a++) {
            var mod = mods_2[_a];
            delete this.attrEqualityMods[mod];
          }
          return this;
        };
        CssSelectorParser3.prototype.enableSubstitutes = function() {
          this.substitutesEnabled = true;
          return this;
        };
        CssSelectorParser3.prototype.disableSubstitutes = function() {
          this.substitutesEnabled = false;
          return this;
        };
        CssSelectorParser3.prototype.parse = function(str) {
          return parser_context_1.parseCssSelector(str, 0, this.pseudos, this.attrEqualityMods, this.ruleNestingOperators, this.substitutesEnabled);
        };
        CssSelectorParser3.prototype.render = function(path) {
          return render_1.renderEntity(path).trim();
        };
        return CssSelectorParser3;
      }()
    );
    exports.CssSelectorParser = CssSelectorParser2;
  }
});

// ../../node_modules/myst-common/node_modules/nanoid/index.browser.js
var random = (bytes) => crypto.getRandomValues(new Uint8Array(bytes));
var customRandom = (alphabet, defaultSize, getRandom) => {
  let mask = (2 << Math.log2(alphabet.length - 1)) - 1;
  let step = -~(1.6 * mask * defaultSize / alphabet.length);
  return (size = defaultSize) => {
    let id = "";
    while (true) {
      let bytes = getRandom(step);
      let j = step | 0;
      while (j--) {
        id += alphabet[bytes[j] & mask] || "";
        if (id.length >= size)
          return id;
      }
    }
  };
};
var customAlphabet = (alphabet, size = 21) => customRandom(alphabet, size | 0, random);

// ../../node_modules/myst-common/dist/utils.js
function addMessageInfo(message, info) {
  if (info === null || info === void 0 ? void 0 : info.note)
    message.note = info.note;
  if (info === null || info === void 0 ? void 0 : info.url)
    message.url = info.url;
  if (info === null || info === void 0 ? void 0 : info.ruleId)
    message.ruleId = info.ruleId;
  if (info === null || info === void 0 ? void 0 : info.key)
    message.key = info.key;
  if (info === null || info === void 0 ? void 0 : info.fatal)
    message.fatal = true;
  return message;
}
function fileError(file, message, opts) {
  return addMessageInfo(file.message(message, opts === null || opts === void 0 ? void 0 : opts.node, opts === null || opts === void 0 ? void 0 : opts.source), { ...opts, fatal: true });
}
function fileWarn(file, message, opts) {
  return addMessageInfo(file.message(message, opts === null || opts === void 0 ? void 0 : opts.node, opts === null || opts === void 0 ? void 0 : opts.source), opts);
}
var az = "abcdefghijklmnopqrstuvwxyz";
var alpha = az + az.toUpperCase();
var numbers = "0123456789";
var nanoidAZ = customAlphabet(alpha, 1);
var nanoidAZ9 = customAlphabet(alpha + numbers, 9);
function createId() {
  return nanoidAZ() + nanoidAZ9();
}
function normalizeLabel(label) {
  if (!label)
    return void 0;
  const identifier = label.replace(/[\t\n\r ]+/g, " ").replace(/['‘’"“”]+/g, "").trim().toLowerCase();
  const html_id = createHtmlId(identifier);
  return { identifier, label, html_id };
}
function createHtmlId(identifier) {
  if (!identifier)
    return void 0;
  return identifier.toLowerCase().replace(/[^a-z0-9-]/g, "-").replace(/^([0-9-])/, "id-$1").replace(/-[-]+/g, "-").replace(/(?:^[-]+)|(?:[-]+$)/g, "");
}
function transferTargetAttrs(sourceNode, destNode, vfile) {
  if (sourceNode.label) {
    if (destNode.label && vfile && destNode.label !== sourceNode.label) {
      fileWarn(vfile, `label "${destNode.label}" replaced with "${sourceNode.label}"`, {
        node: destNode
      });
    }
    if (destNode.label && vfile && destNode.label === sourceNode.label) {
      fileWarn(vfile, `duplicate label "${destNode.label}" replacement`, {
        node: destNode
      });
    }
    destNode.label = sourceNode.label;
    delete sourceNode.label;
  }
  if (sourceNode.identifier) {
    destNode.identifier = sourceNode.identifier;
    delete sourceNode.identifier;
  }
  if (sourceNode.html_id) {
    destNode.html_id = sourceNode.html_id;
    delete sourceNode.html_id;
  }
  if (sourceNode.indexEntries) {
    if (!destNode.indexEntries)
      destNode.indexEntries = [];
    destNode.indexEntries.push(...sourceNode.indexEntries);
    delete sourceNode.indexEntries;
  }
}
function getNodeOrLiftedChildren(node, removeType) {
  if (!node.children)
    return [node];
  const children = node.children.map((child) => getNodeOrLiftedChildren(child, removeType)).flat();
  if (node.type === removeType) {
    if (node && node.children == null)
      delete node.children;
    return children;
  }
  node.children = children;
  return [node];
}
function liftChildren(tree, removeType) {
  if (!tree.children)
    return;
  tree.children = tree.children.map((child) => getNodeOrLiftedChildren(child, removeType)).flat();
}
function setTextAsChild(node, text) {
  node.children = [{ type: "text", value: text }];
}
function toText(content) {
  if (!content)
    return "";
  if (!Array.isArray(content))
    return toText([content]);
  return content.map((n) => {
    if (!n || typeof n === "string")
      return n || "";
    if ("value" in n)
      return n.value;
    if ("children" in n && n.children)
      return toText(n.children);
    return "";
  }).join("");
}
function copyNode(node) {
  return structuredClone(node);
}
function admonitionKindToTitle(kind) {
  const transform = {
    attention: "Attention",
    caution: "Caution",
    danger: "Danger",
    error: "Error",
    important: "Important",
    hint: "Hint",
    note: "Note",
    seealso: "See Also",
    tip: "Tip",
    warning: "Warning"
  };
  return transform[kind] || `Unknown Admonition "${kind}"`;
}
function writeTexLabelledComment(title, commands, commentLength) {
  if (!commands || (commands === null || commands === void 0 ? void 0 : commands.length) === 0)
    return "";
  const len = (commentLength - title.length - 4) / 2;
  const start = "".padEnd(Math.ceil(len), "%");
  const end = "".padEnd(Math.floor(len), "%");
  const titleBlock = `${start}  ${title}  ${end}
`;
  return `${titleBlock}${commands.join("\n")}
`;
}
function getMetadataTags(node) {
  var _a;
  if (!node.data)
    return [];
  const tags = (_a = node.data.tags) !== null && _a !== void 0 ? _a : [];
  Object.entries(node.data).forEach(([key, val]) => {
    if (val === true || typeof val === "string" && val.toLowerCase() === "true") {
      tags.push(key);
    }
  });
  return tags.map((tag) => tag.toLowerCase());
}
function slugToUrl(slug) {
  if (slug == null)
    return void 0;
  return slug.replace(/\.index$/, "").replace(/\./g, "/");
}

// ../../node_modules/zwitch/index.js
var own = {}.hasOwnProperty;
function zwitch(key, options) {
  const settings = options || {};
  function one2(value, ...parameters) {
    let fn = one2.invalid;
    const handlers = one2.handlers;
    if (value && own.call(value, key)) {
      const id = String(value[key]);
      fn = own.call(handlers, id) ? handlers[id] : one2.unknown;
    }
    if (fn) {
      return fn.call(this, value, ...parameters);
    }
  }
  one2.handlers = settings.handlers || {};
  one2.invalid = settings.invalid;
  one2.unknown = settings.unknown;
  return one2;
}

// ../../node_modules/unist-util-select/lib/attribute.js
var handle = zwitch("operator", {
  unknown: unknownOperator,
  // @ts-expect-error: hush.
  invalid: exists,
  handlers: {
    "=": exact,
    "^=": begins,
    "$=": ends,
    "*=": containsString,
    "~=": containsArray
  }
});
function attribute(query, node) {
  let index = -1;
  while (++index < query.attrs.length) {
    if (!handle(query.attrs[index], node))
      return false;
  }
  return true;
}
function exists(query, node) {
  return node[query.name] !== null && node[query.name] !== void 0;
}
function exact(query, node) {
  return exists(query, node) && String(node[query.name]) === query.value;
}
function containsArray(query, node) {
  const value = node[query.name];
  if (value === null || value === void 0)
    return false;
  if (Array.isArray(value) && value.includes(query.value)) {
    return true;
  }
  return String(value) === query.value;
}
function begins(query, node) {
  const value = node[query.name];
  return Boolean(
    query.value && typeof value === "string" && value.slice(0, query.value.length) === query.value
  );
}
function ends(query, node) {
  const value = node[query.name];
  return Boolean(
    query.value && typeof value === "string" && value.slice(-query.value.length) === query.value
  );
}
function containsString(query, node) {
  const value = node[query.name];
  return Boolean(
    query.value && typeof value === "string" && value.includes(query.value)
  );
}
function unknownOperator(query) {
  throw new Error("Unknown operator `" + query.operator + "`");
}

// ../../node_modules/unist-util-select/lib/name.js
function name(query, node) {
  return query.tagName === "*" || query.tagName === node.type;
}

// ../../node_modules/nth-check/lib/esm/parse.js
var whitespace = /* @__PURE__ */ new Set([9, 10, 12, 13, 32]);
var ZERO = "0".charCodeAt(0);
var NINE = "9".charCodeAt(0);
function parse(formula) {
  formula = formula.trim().toLowerCase();
  if (formula === "even") {
    return [2, 0];
  } else if (formula === "odd") {
    return [2, 1];
  }
  let idx = 0;
  let a = 0;
  let sign = readSign();
  let number = readNumber();
  if (idx < formula.length && formula.charAt(idx) === "n") {
    idx++;
    a = sign * (number !== null && number !== void 0 ? number : 1);
    skipWhitespace();
    if (idx < formula.length) {
      sign = readSign();
      skipWhitespace();
      number = readNumber();
    } else {
      sign = number = 0;
    }
  }
  if (number === null || idx < formula.length) {
    throw new Error(`n-th rule couldn't be parsed ('${formula}')`);
  }
  return [a, sign * number];
  function readSign() {
    if (formula.charAt(idx) === "-") {
      idx++;
      return -1;
    }
    if (formula.charAt(idx) === "+") {
      idx++;
    }
    return 1;
  }
  function readNumber() {
    const start = idx;
    let value = 0;
    while (idx < formula.length && formula.charCodeAt(idx) >= ZERO && formula.charCodeAt(idx) <= NINE) {
      value = value * 10 + (formula.charCodeAt(idx) - ZERO);
      idx++;
    }
    return idx === start ? null : value;
  }
  function skipWhitespace() {
    while (idx < formula.length && whitespace.has(formula.charCodeAt(idx))) {
      idx++;
    }
  }
}

// ../../node_modules/nth-check/lib/esm/compile.js
var import_boolbase = __toESM(require_boolbase(), 1);
function compile(parsed) {
  const a = parsed[0];
  const b = parsed[1] - 1;
  if (b < 0 && a <= 0)
    return import_boolbase.default.falseFunc;
  if (a === -1)
    return (index) => index <= b;
  if (a === 0)
    return (index) => index === b;
  if (a === 1)
    return b < 0 ? import_boolbase.default.trueFunc : (index) => index >= b;
  const absA = Math.abs(a);
  const bMod = (b % absA + absA) % absA;
  return a > 1 ? (index) => index >= b && index % absA === bMod : (index) => index <= b && index % absA === bMod;
}

// ../../node_modules/nth-check/lib/esm/index.js
function nthCheck(formula) {
  return compile(parse(formula));
}

// ../../node_modules/unist-util-select/lib/util.js
function parent(node) {
  return Array.isArray(node.children);
}

// ../../node_modules/unist-util-select/lib/pseudo.js
var nthCheck2 = nthCheck.default || nthCheck;
var handle2 = zwitch("name", {
  unknown: unknownPseudo,
  invalid: invalidPseudo,
  handlers: {
    any: matches,
    blank: empty,
    empty,
    "first-child": firstChild,
    "first-of-type": firstOfType,
    has,
    "last-child": lastChild,
    "last-of-type": lastOfType,
    matches,
    not,
    "nth-child": nthChild,
    "nth-last-child": nthLastChild,
    "nth-of-type": nthOfType,
    "nth-last-of-type": nthLastOfType,
    "only-child": onlyChild,
    "only-of-type": onlyOfType,
    root,
    scope
  }
});
pseudo.needsIndex = [
  "any",
  "first-child",
  "first-of-type",
  "last-child",
  "last-of-type",
  "matches",
  "not",
  "nth-child",
  "nth-last-child",
  "nth-of-type",
  "nth-last-of-type",
  "only-child",
  "only-of-type"
];
function pseudo(query, node, index, parent2, state) {
  const pseudos = query.pseudos;
  let offset = -1;
  while (++offset < pseudos.length) {
    if (!handle2(pseudos[offset], node, index, parent2, state))
      return false;
  }
  return true;
}
function empty(_1, node) {
  return parent(node) ? node.children.length === 0 : !("value" in node);
}
function firstChild(query, _1, _2, _3, state) {
  assertDeep(state, query);
  return state.nodeIndex === 0;
}
function firstOfType(query, _1, _2, _3, state) {
  assertDeep(state, query);
  return state.typeIndex === 0;
}
function has(query, node, _1, _2, state) {
  const fragment = { type: "root", children: parent(node) ? node.children : [] };
  const childState = {
    ...state,
    // Not found yet.
    found: false,
    // Do walk deep.
    shallow: false,
    // One result is enough.
    one: true,
    scopeNodes: [node],
    results: [],
    rootQuery: queryToSelectors(query.value)
  };
  walk(childState, fragment);
  return childState.results.length > 0;
}
function lastChild(query, _1, _2, _3, state) {
  assertDeep(state, query);
  return typeof state.nodeCount === "number" && state.nodeIndex === state.nodeCount - 1;
}
function lastOfType(query, _1, _2, _3, state) {
  assertDeep(state, query);
  return typeof state.typeCount === "number" && state.typeIndex === state.typeCount - 1;
}
function matches(query, node, _1, _2, state) {
  const childState = {
    ...state,
    // Not found yet.
    found: false,
    // Do walk deep.
    shallow: false,
    // One result is enough.
    one: true,
    scopeNodes: [node],
    results: [],
    rootQuery: queryToSelectors(query.value)
  };
  walk(childState, node);
  return childState.results[0] === node;
}
function not(query, node, index, parent2, state) {
  return !matches(query, node, index, parent2, state);
}
function nthChild(query, _1, _2, _3, state) {
  const fn = getCachedNthCheck(query);
  assertDeep(state, query);
  return typeof state.nodeIndex === "number" && fn(state.nodeIndex);
}
function nthLastChild(query, _1, _2, _3, state) {
  const fn = getCachedNthCheck(query);
  assertDeep(state, query);
  return typeof state.nodeCount === "number" && typeof state.nodeIndex === "number" && fn(state.nodeCount - state.nodeIndex - 1);
}
function nthLastOfType(query, _1, _2, _3, state) {
  const fn = getCachedNthCheck(query);
  assertDeep(state, query);
  return typeof state.typeIndex === "number" && typeof state.typeCount === "number" && fn(state.typeCount - 1 - state.typeIndex);
}
function nthOfType(query, _1, _2, _3, state) {
  const fn = getCachedNthCheck(query);
  assertDeep(state, query);
  return typeof state.typeIndex === "number" && fn(state.typeIndex);
}
function onlyChild(query, _1, _2, _3, state) {
  assertDeep(state, query);
  return state.nodeCount === 1;
}
function onlyOfType(query, _1, _2, _3, state) {
  assertDeep(state, query);
  return state.typeCount === 1;
}
function root(_1, node, _2, parent2) {
  return node && !parent2;
}
function scope(_1, node, _2, _3, state) {
  return node && state.scopeNodes.includes(node);
}
function invalidPseudo() {
  throw new Error("Invalid pseudo-selector");
}
function unknownPseudo(query) {
  if (query.name) {
    throw new Error("Unknown pseudo-selector `" + query.name + "`");
  }
  throw new Error("Unexpected pseudo-element or empty pseudo-class");
}
function assertDeep(state, query) {
  if (state.shallow) {
    throw new Error("Cannot use `:" + query.name + "` without parent");
  }
}
function getCachedNthCheck(query) {
  let fn = query._cachedFn;
  if (!fn) {
    fn = nthCheck2(query.value);
    query._cachedFn = fn;
  }
  return fn;
}

// ../../node_modules/unist-util-select/lib/test.js
function test(query, node, index, parent2, state) {
  if (query.id)
    throw new Error("Invalid selector: id");
  if (query.classNames)
    throw new Error("Invalid selector: class");
  return Boolean(
    node && (!query.tagName || name(query, node)) && (!query.attrs || attribute(query, node)) && (!query.pseudos || pseudo(query, node, index, parent2, state))
  );
}

// ../../node_modules/unist-util-select/lib/walk.js
var empty2 = [];
function queryToSelectors(query) {
  if (query === null) {
    return { type: "selectors", selectors: [] };
  }
  if (query.type === "ruleSet") {
    return { type: "selectors", selectors: [query] };
  }
  return query;
}
function walk(state, tree) {
  if (tree) {
    one(state, [], tree, void 0, void 0);
  }
}
function one(state, currentRules, node, index, parentNode) {
  let nestResult = {
    directChild: void 0,
    descendant: void 0,
    adjacentSibling: void 0,
    generalSibling: void 0
  };
  nestResult = applySelectors(
    state,
    // Try the root rules for this node too.
    combine(currentRules, state.rootQuery.selectors),
    node,
    index,
    parentNode
  );
  if (parent(node) && !state.shallow && !(state.one && state.found)) {
    all(state, nestResult, node);
  }
  return nestResult;
}
function all(state, nest, node) {
  const fromParent = combine(nest.descendant, nest.directChild);
  let fromSibling;
  let index = -1;
  const total = { count: 0, types: /* @__PURE__ */ new Map() };
  const before = { count: 0, types: /* @__PURE__ */ new Map() };
  while (++index < node.children.length) {
    count(total, node.children[index]);
  }
  index = -1;
  while (++index < node.children.length) {
    const child = node.children[index];
    const name2 = child.type.toUpperCase();
    state.nodeIndex = before.count;
    state.typeIndex = before.types.get(name2) || 0;
    state.nodeCount = total.count;
    state.typeCount = total.types.get(name2);
    const forSibling = combine(fromParent, fromSibling);
    const nest2 = one(state, forSibling, node.children[index], index, node);
    fromSibling = combine(nest2.generalSibling, nest2.adjacentSibling);
    if (state.one && state.found) {
      break;
    }
    count(before, node.children[index]);
  }
}
function applySelectors(state, rules, node, index, parent2) {
  const nestResult = {
    directChild: void 0,
    descendant: void 0,
    adjacentSibling: void 0,
    generalSibling: void 0
  };
  let selectorIndex = -1;
  while (++selectorIndex < rules.length) {
    const ruleSet = rules[selectorIndex];
    if (state.one && state.found) {
      break;
    }
    if (state.shallow && ruleSet.rule.rule) {
      throw new Error("Expected selector without nesting");
    }
    if (test(ruleSet.rule, node, index, parent2, state)) {
      const nest = ruleSet.rule.rule;
      if (nest) {
        const rule = { type: "ruleSet", rule: nest };
        const label = nest.nestingOperator === "+" ? "adjacentSibling" : nest.nestingOperator === "~" ? "generalSibling" : nest.nestingOperator === ">" ? "directChild" : "descendant";
        add(nestResult, label, rule);
      } else {
        state.found = true;
        if (!state.results.includes(node)) {
          state.results.push(node);
        }
      }
    }
    if (ruleSet.rule.nestingOperator === null) {
      add(nestResult, "descendant", ruleSet);
    } else if (ruleSet.rule.nestingOperator === "~") {
      add(nestResult, "generalSibling", ruleSet);
    }
  }
  return nestResult;
}
function combine(left, right) {
  return left && right && left.length > 0 && right.length > 0 ? [...left, ...right] : left && left.length > 0 ? left : right && right.length > 0 ? right : empty2;
}
function add(nest, field, rule) {
  const list = nest[field];
  if (list) {
    list.push(rule);
  } else {
    nest[field] = [rule];
  }
}
function count(counts, node) {
  const name2 = node.type.toUpperCase();
  const count2 = (counts.types.get(name2) || 0) + 1;
  counts.count++;
  counts.types.set(name2, count2);
}

// ../../node_modules/unist-util-select/lib/parse.js
var import_css_selector_parser = __toESM(require_lib(), 1);
var parser = new import_css_selector_parser.CssSelectorParser();
parser.registerAttrEqualityMods("~", "^", "$", "*");
parser.registerSelectorPseudos("any", "matches", "not", "has");
parser.registerNestingOperators(">", "+", "~");
function parse2(selector) {
  if (typeof selector !== "string") {
    throw new TypeError("Expected `string` as selector, not `" + selector + "`");
  }
  return parser.parse(selector);
}

// ../../node_modules/unist-util-select/index.js
function matches2(selector, node) {
  const state = createState(selector, node);
  state.one = true;
  state.shallow = true;
  walk(state, node || void 0);
  return state.results.length > 0;
}
function select(selector, tree) {
  const state = createState(selector, tree);
  state.one = true;
  walk(state, tree || void 0);
  return state.results[0] || null;
}
function selectAll(selector, tree) {
  const state = createState(selector, tree);
  walk(state, tree || void 0);
  return state.results;
}
function createState(selector, tree) {
  return {
    // State of the query.
    rootQuery: queryToSelectors(parse2(selector)),
    results: [],
    scopeNodes: tree ? parent(tree) && // Root in nlcst.
    (tree.type === "RootNode" || tree.type === "root") ? tree.children : [tree] : [],
    one: false,
    shallow: false,
    found: false,
    // State in the tree.
    typeIndex: void 0,
    nodeIndex: void 0,
    typeCount: void 0,
    nodeCount: void 0
  };
}

// ../../node_modules/unist-util-remove/lib/index.js
var empty3 = [];
var remove = (
  /**
   * @type {(
   *  (<Tree extends Node>(node: Tree, options: Options, test: Test) => Tree | null) &
   *  (<Tree extends Node>(node: Tree, test: Test) => Tree | null)
   * )}
   */
  /**
   * @param {Node} tree
   * @param {Options | null | undefined} [options]
   * @param {Test | null | undefined} [test]
   * @returns {Node | null}
   */
  function(tree, options, test2) {
    const is = convert(test2 || options);
    const cascade = !options || options.cascade === void 0 || options.cascade === null ? true : options.cascade;
    return preorder(tree);
    function preorder(node, index, parent2) {
      const children = node.children || empty3;
      let childIndex = -1;
      let position = 0;
      if (is(node, index, parent2)) {
        return null;
      }
      if (children.length > 0) {
        while (++childIndex < children.length) {
          if (preorder(children[childIndex], childIndex, node)) {
            children[position++] = children[childIndex];
          }
        }
        if (cascade && !position) {
          return null;
        }
        children.length = position;
      }
      return node;
    }
  }
);

// ../../node_modules/myst-frontmatter/dist/site/types.js
var PAGE_KNOWN_PARTS = [
  "abstract",
  "summary",
  "keypoints",
  "dedication",
  "epigraph",
  "data_availability",
  "acknowledgments"
];
var SITE_FRONTMATTER_KEYS = [
  "title",
  "subtitle",
  "short_title",
  "description",
  "thumbnail",
  "thumbnailOptimized",
  "banner",
  "bannerOptimized",
  "tags",
  "authors",
  "reviewers",
  "editors",
  "contributors",
  "venue",
  "github",
  "keywords",
  "affiliations",
  "funding",
  "copyright",
  "options",
  "parts",
  "social",
  ...PAGE_KNOWN_PARTS
];
var FRONTMATTER_ALIASES = {
  author: "authors",
  reviewer: "reviewers",
  editor: "editors",
  contributor: "contributors",
  affiliation: "affiliations",
  export: "exports",
  download: "downloads",
  jupyter: "thebe",
  part: "parts",
  ack: "acknowledgments",
  acknowledgements: "acknowledgments",
  acknowledgment: "acknowledgments",
  acknowledgement: "acknowledgments",
  availability: "data_availability",
  dataAvailability: "data_availability",
  "data-availability": "data_availability",
  quote: "epigraph",
  plain_language_summary: "summary",
  "plain-language-summary": "summary",
  plainLanguageSummary: "summary",
  lay_summary: "summary",
  "lay-summary": "summary",
  keyPoints: "keypoints",
  key_points: "keypoints",
  "key-points": "keypoints",
  image: "thumbnail",
  identifier: "identifiers",
  socials: "social"
};

// ../../node_modules/myst-common/dist/extractParts.js
function coercePart(part) {
  if (!part) {
    return [];
  }
  if (typeof part === "string")
    return coercePart([part]);
  const parts = [];
  part.map((p) => p.toLowerCase()).forEach((p) => {
    parts.push(p);
    Object.entries(FRONTMATTER_ALIASES).forEach(([alias, value]) => {
      if (p === alias || p === value) {
        if (!parts.includes(value))
          parts.unshift(value);
        if (!parts.includes(alias))
          parts.push(alias);
      }
    });
  });
  return parts;
}
function selectBlockParts(tree, part) {
  const parts = coercePart(part);
  if (parts.length === 0)
    return [];
  const blockParts = selectAll("block", tree).filter((block) => {
    var _a, _b, _c;
    const blockTags = (((_a = block.data) === null || _a === void 0 ? void 0 : _a.tags) && Array.isArray(block.data.tags) ? block.data.tags : []).map((tag) => tag === null || tag === void 0 ? void 0 : tag.toLowerCase());
    const blockPart = (_c = (_b = block.data) === null || _b === void 0 ? void 0 : _b.part) === null || _c === void 0 ? void 0 : _c.toLowerCase();
    return parts.map((p) => blockPart === p || blockTags.includes(p)).reduce((a, b) => a || b, false);
  });
  return blockParts;
}
function selectFrontmatterParts(frontmatterParts, part) {
  if (!frontmatterParts)
    return [];
  const parts = coercePart(part);
  if (parts.length === 0)
    return [];
  const blockParts = [];
  parts.forEach((p) => {
    Object.entries(frontmatterParts).forEach(([key, value]) => {
      if (p === key.toLowerCase())
        blockParts.push(...value.mdast.children);
    });
  });
  return blockParts;
}
function createPartBlock(children, part, opts) {
  var _a;
  const block = { type: "block", children };
  if (!(opts === null || opts === void 0 ? void 0 : opts.removePartData)) {
    (_a = block.data) !== null && _a !== void 0 ? _a : block.data = {};
    block.data.part = part;
  }
  return block;
}
function forcedRemove(tree, test2) {
  let success = remove(tree, test2);
  if (!success) {
    success = remove(tree, { cascade: false }, test2);
  }
  return success;
}
function extractImplicitPart(tree, part, opts) {
  var _a;
  const parts = coercePart(part);
  if (parts.length === 0)
    return;
  let insideImplicitPart = false;
  const blockParts = [];
  let paragraphs = [];
  (_a = tree.children) === null || _a === void 0 ? void 0 : _a.forEach((child, index) => {
    var _a2;
    if (insideImplicitPart && child.type === "paragraph") {
      paragraphs.push(copyNode(child));
      child.type = "__part_delete__";
    }
    if (child.type !== "__part_delete__" || index === tree.children.length - 1) {
      insideImplicitPart = false;
      if (paragraphs.length > 0) {
        blockParts.push(createPartBlock(paragraphs, parts[0], opts));
        paragraphs = [];
        selectAll("__part_heading__", tree).forEach((node) => {
          node.type = "__part_delete__";
        });
      }
    }
    if (child.type === "block") {
      if ((_a2 = child.data) === null || _a2 === void 0 ? void 0 : _a2.part)
        return;
      if (tree.type !== "root")
        return;
      const blockPartsTree = extractImplicitPart(child, parts);
      if (blockPartsTree)
        blockParts.push(...blockPartsTree.children);
    } else if (child.type === "heading" && parts.includes(toText(child).toLowerCase())) {
      insideImplicitPart = true;
      child.type = "__part_heading__";
    }
  });
  selectAll("__part_heading__", tree).forEach((node) => {
    node.type = "heading";
  });
  if (blockParts.length === 0)
    return;
  const partsTree = { type: "root", children: blockParts };
  forcedRemove(tree, "__part_delete__");
  return partsTree;
}
function extractPart(tree, part, opts) {
  const partStrings = coercePart(part);
  if (partStrings.length === 0)
    return;
  const frontmatterParts = selectFrontmatterParts(opts === null || opts === void 0 ? void 0 : opts.frontmatterParts, part);
  const blockParts = selectBlockParts(tree, part);
  if (frontmatterParts.length === 0 && blockParts.length === 0) {
    if (opts === null || opts === void 0 ? void 0 : opts.requireExplicitPart)
      return;
    return extractImplicitPart(tree, partStrings);
  }
  const children = copyNode(frontmatterParts.length > 0 ? frontmatterParts : blockParts).map((block) => {
    var _a;
    (_a = block.data) !== null && _a !== void 0 ? _a : block.data = {};
    block.data.part = partStrings[0];
    if (block.data.tags && Array.isArray(block.data.tags) && block.data.tags.reduce((a, t) => a || partStrings.includes(t.toLowerCase()), false)) {
      block.data.tags = block.data.tags.filter((tag) => !partStrings.includes(tag.toLowerCase()));
      if (block.data.tags.length === 0) {
        delete block.data.tags;
      }
    }
    if (opts === null || opts === void 0 ? void 0 : opts.removePartData)
      delete block.data.part;
    if (!(opts === null || opts === void 0 ? void 0 : opts.keepVisibility))
      delete block.visibility;
    return block;
  });
  const partsTree = { type: "root", children };
  blockParts.forEach((block) => {
    block.type = "__delete__";
  });
  forcedRemove(tree, "__delete__");
  return partsTree;
}

// ../../node_modules/myst-common/dist/ruleids.js
var RuleId;
(function(RuleId2) {
  RuleId2["validConfigStructure"] = "valid-config-structure";
  RuleId2["siteConfigExists"] = "site-config-exists";
  RuleId2["projectConfigExists"] = "project-config-exists";
  RuleId2["validSiteConfig"] = "valid-site-config";
  RuleId2["validProjectConfig"] = "valid-project-config";
  RuleId2["configHasNoDeprecatedFields"] = "config-has-no-deprecated-fields";
  RuleId2["frontmatterIsYaml"] = "frontmatter-is-yaml";
  RuleId2["validPageFrontmatter"] = "valid-page-frontmatter";
  RuleId2["validFrontmatterExportList"] = "valid-frontmatter-export-list";
  RuleId2["docxRenders"] = "docx-renders";
  RuleId2["jatsRenders"] = "jats-renders";
  RuleId2["mdRenders"] = "md-renders";
  RuleId2["mecaIncludesJats"] = "meca-includes-jats";
  RuleId2["mecaExportsBuilt"] = "meca-exports-built";
  RuleId2["mecaFilesCopied"] = "meca-files-copied";
  RuleId2["pdfBuildCommandsAvailable"] = "pdf-build-commands-available";
  RuleId2["pdfBuildsWithoutErrors"] = "pdf-builds-without-errors";
  RuleId2["pdfBuilds"] = "pdf-builds";
  RuleId2["texRenders"] = "tex-renders";
  RuleId2["exportExtensionCorrect"] = "export-extension-correct";
  RuleId2["exportArticleExists"] = "export-article-exists";
  RuleId2["texParses"] = "tex-parses";
  RuleId2["jatsParses"] = "jats-parses";
  RuleId2["mystFileLoads"] = "myst-file-loads";
  RuleId2["selectedFileIsProcessed"] = "selected-file-is-processed";
  RuleId2["directiveRegistered"] = "directive-registered";
  RuleId2["directiveKnown"] = "directive-known";
  RuleId2["directiveArgumentCorrect"] = "directive-argument-correct";
  RuleId2["directiveOptionsCorrect"] = "directive-options-correct";
  RuleId2["directiveBodyCorrect"] = "directive-body-correct";
  RuleId2["roleRegistered"] = "role-registered";
  RuleId2["roleKnown"] = "role-known";
  RuleId2["roleBodyCorrect"] = "role-body-correct";
  RuleId2["tocContentsExist"] = "toc-contents-exist";
  RuleId2["encounteredLegacyTOC"] = "encountered-legacy-toc";
  RuleId2["validTOCStructure"] = "valid-toc-structure";
  RuleId2["validTOC"] = "valid-toc";
  RuleId2["imageDownloads"] = "image-downloads";
  RuleId2["imageExists"] = "image-exists";
  RuleId2["imageFormatConverts"] = "image-format-converts";
  RuleId2["imageCopied"] = "image-copied";
  RuleId2["imageFormatOptimizes"] = "image-format-optimizes";
  RuleId2["mathLabelLifted"] = "math-label-lifted";
  RuleId2["mathEquationEnvRemoved"] = "math-equation-env-removed";
  RuleId2["mathEqnarrayReplaced"] = "math-eqnarray-replaced";
  RuleId2["mathAlignmentAdjusted"] = "math-alignment-adjusted";
  RuleId2["mathRenders"] = "math-renders";
  RuleId2["referenceTemplateFills"] = "reference-template-fills";
  RuleId2["identifierIsUnique"] = "identifier-is-unique";
  RuleId2["referenceTargetResolves"] = "reference-target-resolves";
  RuleId2["referenceSyntaxValid"] = "reference-syntax-valid";
  RuleId2["referenceTargetExplicit"] = "reference-target-explicit";
  RuleId2["footnoteReferencesDefinition"] = "footnote-references-definition";
  RuleId2["intersphinxReferencesResolve"] = "intersphinx-references-resolve";
  RuleId2["mystLinkValid"] = "myst-link-valid";
  RuleId2["sphinxLinkValid"] = "sphinx-link-valid";
  RuleId2["rridLinkValid"] = "rrid-link-valid";
  RuleId2["rorLinkValid"] = "ror-link-valid";
  RuleId2["wikipediaLinkValid"] = "wikipedia-link-valid";
  RuleId2["doiLinkValid"] = "doi-link-valid";
  RuleId2["linkResolves"] = "link-resolves";
  RuleId2["linkTextExists"] = "link-text-exists";
  RuleId2["notebookAttachmentsResolve"] = "notebook-attachments-resolve";
  RuleId2["notebookOutputCopied"] = "notebook-output-copied";
  RuleId2["mdastSnippetImports"] = "mdast-snippet-imports";
  RuleId2["includeContentFilters"] = "include-content-filters";
  RuleId2["includeContentLoads"] = "include-content-loads";
  RuleId2["gatedNodesJoin"] = "gated-nodes-join";
  RuleId2["glossaryUsesDefinitionList"] = "glossary-uses-definition-list";
  RuleId2["blockMetadataLoads"] = "block-metadata-loads";
  RuleId2["indexEntriesResolve"] = "index-entries-resolve";
  RuleId2["citationIsUnique"] = "citation-is-unique";
  RuleId2["bibFileExists"] = "bib-file-exists";
  RuleId2["citationRenders"] = "citation-renders";
  RuleId2["codeMetadataLifted"] = "code-metadata-lifted";
  RuleId2["codeMetatagsValid"] = "code-metatags-valid";
  RuleId2["codeLangDefined"] = "code-lang-defined";
  RuleId2["codeMetadataLoads"] = "code-metadata-loads";
  RuleId2["inlineCodeMalformed"] = "inline-code-malformed";
  RuleId2["inlineExpressionRenders"] = "inline-expression-renders";
  RuleId2["staticFileCopied"] = "static-file-copied";
  RuleId2["exportFileCopied"] = "export-file-copied";
  RuleId2["sourceFileCopied"] = "source-file-copied";
  RuleId2["templateFileCopied"] = "template-file-copied";
  RuleId2["staticActionFileCopied"] = "static-action-file-copied";
  RuleId2["pluginLoads"] = "plugin-loads";
  RuleId2["containerChildrenValid"] = "container-children-valid";
  RuleId2["mystJsonValid"] = "myst-json-valid";
  RuleId2["codeCellExecutes"] = "code-cell-executes";
  RuleId2["inlineExpressionExecutes"] = "inline-expression-executes";
})(RuleId || (RuleId = {}));
var RULE_ID_DESCRIPTIONS = {
  // Frontmatter rules
  [RuleId.validConfigStructure]: "Configuration file structure is valid and can be parsed",
  [RuleId.siteConfigExists]: "Site configuration is found in project",
  [RuleId.projectConfigExists]: "Project configuration file exists in the directory",
  [RuleId.validSiteConfig]: "Site configuration passes validation",
  [RuleId.validProjectConfig]: "Project configuration passes validation",
  [RuleId.configHasNoDeprecatedFields]: "Configuration uses current field names without deprecated options",
  [RuleId.frontmatterIsYaml]: "Frontmatter can be parsed as valid YAML",
  [RuleId.validPageFrontmatter]: "Page frontmatter passes validation",
  [RuleId.validFrontmatterExportList]: "Export list in frontmatter is valid for the specified format",
  // Export rules
  [RuleId.docxRenders]: "DOCX document renders without errors",
  [RuleId.jatsRenders]: "JATS XML renders without errors",
  [RuleId.mdRenders]: "Markdown output renders without errors",
  [RuleId.mecaIncludesJats]: "MECA bundle contains required JATS file",
  [RuleId.mecaExportsBuilt]: "MECA archive builds without errors",
  [RuleId.mecaFilesCopied]: "MECA files copy to output successfully",
  [RuleId.pdfBuildCommandsAvailable]: "Required PDF build tools (LaTeX/Typst) are installed",
  [RuleId.pdfBuildsWithoutErrors]: "PDF compilation completes without LaTeX/Typst errors",
  [RuleId.pdfBuilds]: "PDF file generates successfully",
  [RuleId.texRenders]: "LaTeX/Typst document renders without errors",
  [RuleId.exportExtensionCorrect]: "Export file has the correct extension for its format",
  [RuleId.exportArticleExists]: "Article file specified in export configuration exists",
  // Parse rules
  [RuleId.texParses]: "LaTeX content parses without syntax errors",
  [RuleId.jatsParses]: "JATS XML parses without syntax errors",
  [RuleId.mystFileLoads]: "MyST markdown file loads and parses successfully",
  [RuleId.selectedFileIsProcessed]: "File specified for processing is found and processed",
  // Directive and role rules
  [RuleId.directiveRegistered]: "Directive is registered without name conflicts",
  [RuleId.directiveKnown]: "Directive name is recognized",
  [RuleId.directiveArgumentCorrect]: "Directive argument matches specification",
  [RuleId.directiveOptionsCorrect]: "Directive options are valid and properly formatted",
  [RuleId.directiveBodyCorrect]: "Directive body content meets requirements",
  [RuleId.roleRegistered]: "Role is registered without name conflicts",
  [RuleId.roleKnown]: "Role name is recognized",
  [RuleId.roleBodyCorrect]: "Role content meets requirements",
  // Project structure rules
  [RuleId.tocContentsExist]: "Files referenced in table of contents exist",
  [RuleId.encounteredLegacyTOC]: "Table of contents uses deprecated format",
  [RuleId.validTOCStructure]: "Table of contents structure passes schema validation",
  [RuleId.validTOC]: "Table of contents is valid",
  // Image rules
  [RuleId.imageDownloads]: "Remote image downloads successfully",
  [RuleId.imageExists]: "Image file exists at specified path",
  [RuleId.imageFormatConverts]: "Image format converts successfully using available tools",
  [RuleId.imageCopied]: "Image copies to output directory successfully",
  [RuleId.imageFormatOptimizes]: "Image format optimizes successfully",
  // Math rules
  [RuleId.mathLabelLifted]: "Math equation label extracts successfully",
  [RuleId.mathEquationEnvRemoved]: "Nested equation environment removes successfully",
  [RuleId.mathEqnarrayReplaced]: "Deprecated eqnarray environment replaces with align",
  [RuleId.mathAlignmentAdjusted]: "Math alignment adjusts successfully",
  [RuleId.mathRenders]: "Math expression renders with KaTeX without errors",
  // Reference rules
  [RuleId.referenceTemplateFills]: "Reference template populates with values successfully",
  [RuleId.identifierIsUnique]: "Identifier is unique across the project",
  [RuleId.referenceTargetResolves]: "Cross-reference target is found",
  [RuleId.referenceSyntaxValid]: "Reference syntax is valid",
  [RuleId.referenceTargetExplicit]: "Reference target is explicitly specified",
  [RuleId.footnoteReferencesDefinition]: "Footnote reference links to an existing definition",
  [RuleId.intersphinxReferencesResolve]: "Intersphinx cross-reference resolves to external documentation",
  // Link rules
  [RuleId.mystLinkValid]: "MyST-formatted link is valid",
  [RuleId.sphinxLinkValid]: "Sphinx-style link is valid",
  [RuleId.rridLinkValid]: "RRID (Research Resource Identifier) link is valid",
  [RuleId.rorLinkValid]: "ROR (Research Organization Registry) link is valid",
  [RuleId.wikipediaLinkValid]: "Wikipedia link is valid",
  [RuleId.doiLinkValid]: "DOI (Digital Object Identifier) link is valid",
  [RuleId.linkResolves]: "Link URL resolves successfully",
  [RuleId.linkTextExists]: "Link has non-empty text content",
  // Notebook rules
  [RuleId.notebookAttachmentsResolve]: "Jupyter notebook attachments resolve and decode successfully",
  [RuleId.notebookOutputCopied]: "Notebook cell output copies to disk successfully",
  // Content rules
  [RuleId.mdastSnippetImports]: "MDAST snippet imports successfully",
  [RuleId.includeContentFilters]: "Include directive filters apply correctly",
  [RuleId.includeContentLoads]: "Include directive loads target file successfully without circular dependencies",
  [RuleId.gatedNodesJoin]: "Conditional content nodes merge correctly",
  [RuleId.glossaryUsesDefinitionList]: "Glossary directive contains a definition list",
  [RuleId.blockMetadataLoads]: "Block-level metadata loads and parses successfully",
  [RuleId.indexEntriesResolve]: "Index entries resolve successfully",
  // Citation rules
  [RuleId.citationIsUnique]: "Citation identifier is unique across bibliography files",
  [RuleId.bibFileExists]: "Bibliography file specified in configuration exists",
  [RuleId.citationRenders]: "Citation processes with citation-js successfully",
  // Code rules
  [RuleId.codeMetadataLifted]: "Code cell metadata extracts without conflicts",
  [RuleId.codeMetatagsValid]: "Code cell tags are valid",
  [RuleId.codeLangDefined]: "Code block has a language specified",
  [RuleId.codeMetadataLoads]: "Code metadata loads successfully",
  [RuleId.inlineCodeMalformed]: "Inline code has either value or children but not both",
  [RuleId.inlineExpressionRenders]: "Inline expression evaluates successfully",
  // Static file rules
  [RuleId.staticFileCopied]: "Static file copies to output successfully",
  [RuleId.exportFileCopied]: "Export output file copies to public directory successfully",
  [RuleId.sourceFileCopied]: "Source file copies to output successfully",
  [RuleId.templateFileCopied]: "Template file copies successfully",
  [RuleId.staticActionFileCopied]: "Static action file copies successfully",
  // Plugins
  [RuleId.pluginLoads]: "Plugin loads and executes without errors",
  // Container rules
  [RuleId.containerChildrenValid]: "Container has valid child elements",
  // File rules
  [RuleId.mystJsonValid]: "MyST JSON file is valid",
  [RuleId.codeCellExecutes]: "Code cell executes without errors",
  [RuleId.inlineExpressionExecutes]: "Inline expression evaluates without errors"
};

// ../../node_modules/myst-common/dist/plural.js
function plural(f, count2) {
  var _a;
  const num = (_a = typeof count2 === "number" ? count2 : Array.isArray(count2) ? count2 === null || count2 === void 0 ? void 0 : count2.length : Object.keys(count2 !== null && count2 !== void 0 ? count2 : {}).length) !== null && _a !== void 0 ? _a : 0;
  return f.replace("%s", String(num)).replace(/\((?:([a-z0-9A-Z-]*)\|)?([a-z0-9A-Z-]*)\)/g, num === 1 ? "$1" : "$2");
}

// ../../node_modules/myst-common/dist/indices.js
function parseIndexLine(line, { single, pair, triple, see, seealso }, vfile, node) {
  if (line.trim().length === 0)
    return;
  const splitLine = line.split(/(?<!\\):/).map((val) => val.trim().replace("\\:", ":"));
  if (splitLine.length > 2) {
    fileError(vfile, `Too many colons encountered in index line "${line}"`, {
      node,
      note: 'Index entry must follow pattern "type: entry; sub entry"'
    });
  } else if (splitLine.length === 2) {
    const [entryType, entryValue] = splitLine;
    if (entryType === "single")
      single.push(entryValue);
    else if (entryType === "pair")
      pair.push(entryValue);
    else if (entryType === "triple")
      triple.push(entryValue);
    else if (entryType === "see")
      see.push(entryValue);
    else if (entryType === "seealso")
      seealso.push(entryValue);
    else {
      fileError(vfile, `Unknown index entry type "${entryType}"`, {
        node,
        note: "Allowed types include: single, pair, triple, see, and seealso"
      });
    }
  } else {
    single.push(...splitLine[0].split(/(?<!\\),/).map((val) => val.trim().replace("\\,", ",")));
  }
}
function splitEntryValue(entry) {
  const emphasis = entry.startsWith("!");
  const splitEntry = entry.replace(/^!/, "").replace(/^\\!/, "!").split(/(?<!\\);/).map((val) => val.trim().replace("\\;", ";")).filter((val) => val !== "");
  return { emphasis, splitEntry };
}
function createIndexEntry(entry, sub, emphasis, kind) {
  const subEntry = sub ? { value: sub, kind: kind !== null && kind !== void 0 ? kind : "entry" } : void 0;
  return { entry, subEntry, emphasis };
}
function createIndexEntries({ single, pair, triple, see, seealso }, vfile, node) {
  const entries = [];
  single.forEach((singleEntry) => {
    const { emphasis, splitEntry } = splitEntryValue(singleEntry);
    if (splitEntry.length !== 1 && splitEntry.length !== 2) {
      fileError(vfile, `Unable to parse index "single" entry "${singleEntry}"`, {
        node,
        note: 'Single index entry must follow pattern "entry" or "entry; sub entry"'
      });
    } else {
      const [entry, subEntry] = splitEntry;
      entries.push(createIndexEntry(entry, subEntry, emphasis));
    }
  });
  pair.forEach((pairEntry) => {
    const { emphasis, splitEntry } = splitEntryValue(pairEntry);
    if (splitEntry.length !== 2) {
      fileError(vfile, `Unable to parse index "pair" entry "${pairEntry}"`, {
        node,
        note: 'Pair index entry must follow pattern "entry; sub entry"'
      });
    } else {
      const [entry, subEntry] = splitEntry;
      entries.push(createIndexEntry(entry, subEntry, emphasis));
      entries.push(createIndexEntry(subEntry, entry, emphasis));
    }
  });
  triple.forEach((tripleEntry) => {
    const { emphasis, splitEntry } = splitEntryValue(tripleEntry);
    if (splitEntry.length !== 3) {
      fileError(vfile, `Unable to parse index "triple" entry "${tripleEntry}"`, {
        node,
        note: 'Triple index entry must follow pattern "entry one; entry two; entry three"'
      });
    } else {
      entries.push(createIndexEntry(splitEntry[0], splitEntry[1], emphasis));
      entries.push(createIndexEntry(splitEntry[1], splitEntry[0], emphasis));
      entries.push(createIndexEntry(splitEntry[1], splitEntry[2], emphasis));
      entries.push(createIndexEntry(splitEntry[2], splitEntry[1], emphasis));
      entries.push(createIndexEntry(splitEntry[0], splitEntry[2], emphasis));
      entries.push(createIndexEntry(splitEntry[2], splitEntry[0], emphasis));
    }
  });
  const addSeeEntry = (seeEntry, kind) => {
    const { emphasis, splitEntry } = splitEntryValue(seeEntry);
    if (splitEntry.length !== 2) {
      fileError(vfile, `Unable to parse index "${kind}" entry "${seeEntry}"`, {
        node,
        note: 'See index entry must follow pattern "entry; sub entry"'
      });
    } else {
      const [entry, subEntry] = splitEntry;
      entries.push(createIndexEntry(entry, subEntry, emphasis, kind));
    }
  };
  see.forEach((seeEntry) => {
    addSeeEntry(seeEntry, "see");
  });
  seealso.forEach((seealsoEntry) => {
    addSeeEntry(seealsoEntry, "seealso");
  });
  if (entries.length === 0) {
    fileError(vfile, "No entries parsed from index directive", { node });
  }
  return entries;
}

// ../../node_modules/myst-common/dist/selectNodes.js
function isTargetIdentifierNode(node) {
  const nonTargetTypes = [
    "crossReference",
    "cite",
    "footnoteDefinition",
    "footnoteReference",
    "captionNumber",
    "link"
  ];
  return !nonTargetTypes.includes(node.type);
}
var hiddenNodes = /* @__PURE__ */ new Set(["comment", "mystComment"]);
function selectHeadingNodes(mdast, identifier, maxNodes) {
  let begin = false;
  let htmlId = void 0;
  const nodes = [];
  visit(mdast, (node) => {
    if (begin && node.type === "heading" || maxNodes && nodes.length >= maxNodes) {
      return EXIT;
    }
    if (node.identifier === identifier && node.type === "heading") {
      begin = true;
      htmlId = node.html_id || node.identifier;
    }
    if (begin) {
      if (!hiddenNodes.has(node.type))
        nodes.push(node);
      return SKIP;
    }
  });
  return { htmlId, nodes };
}
function selectDefinitionTerm(mdast, identifier, maxNodes) {
  var _a, _b;
  let begin = false;
  const nodes = [];
  visit(mdast, (node) => {
    if (begin && node.type === "definitionTerm") {
      if (nodes.length > 1)
        return EXIT;
    } else if (begin && node.type !== "definitionDescription") {
      return EXIT;
    }
    if (node.identifier === identifier && node.type === "definitionTerm") {
      nodes.push(node);
      begin = true;
    }
    if (begin) {
      if (node.type === "definitionDescription")
        nodes.push(node);
      return SKIP;
    }
  });
  return {
    htmlId: ((_a = nodes === null || nodes === void 0 ? void 0 : nodes[0]) === null || _a === void 0 ? void 0 : _a.html_id) || ((_b = nodes === null || nodes === void 0 ? void 0 : nodes[0]) === null || _b === void 0 ? void 0 : _b.identifier),
    nodes: [{ type: "definitionList", key: "dl", children: nodes.slice(0, maxNodes) }]
  };
}
function selectMdastNodes(mdast, identifier, maxNodes) {
  if (maxNodes === 0)
    return { nodes: [] };
  const node = selectAll(`[identifier=${identifier}],[key=${identifier}]`, mdast).find((n) => isTargetIdentifierNode(n));
  if (!node)
    return { nodes: [] };
  switch (node.type) {
    case "heading":
      return selectHeadingNodes(mdast, identifier, maxNodes);
    case "definitionTerm":
      return selectDefinitionTerm(mdast, identifier, maxNodes);
    default:
      return { htmlId: node.html_id || node.identifier, nodes: [node] };
  }
}

// ../../node_modules/myst-common/dist/types.js
var NotebookCell;
(function(NotebookCell2) {
  NotebookCell2["content"] = "notebook-content";
  NotebookCell2["code"] = "notebook-code";
})(NotebookCell || (NotebookCell = {}));
var NotebookCellTags;
(function(NotebookCellTags2) {
  NotebookCellTags2["removeStderr"] = "remove-stderr";
  NotebookCellTags2["removeStdout"] = "remove-stdout";
  NotebookCellTags2["hideCell"] = "hide-cell";
  NotebookCellTags2["hideInput"] = "hide-input";
  NotebookCellTags2["hideOutput"] = "hide-output";
  NotebookCellTags2["removeCell"] = "remove-cell";
  NotebookCellTags2["removeInput"] = "remove-input";
  NotebookCellTags2["removeOutput"] = "remove-output";
  NotebookCellTags2["scrollOutput"] = "scroll-output";
  NotebookCellTags2["skipExecution"] = "skip-execution";
  NotebookCellTags2["raisesException"] = "raises-exception";
})(NotebookCellTags || (NotebookCellTags = {}));
var ParseTypesEnum;
(function(ParseTypesEnum2) {
  ParseTypesEnum2["string"] = "string";
  ParseTypesEnum2["number"] = "number";
  ParseTypesEnum2["boolean"] = "boolean";
  ParseTypesEnum2["parsed"] = "parsed";
})(ParseTypesEnum || (ParseTypesEnum = {}));
var TargetKind;
(function(TargetKind2) {
  TargetKind2["heading"] = "heading";
  TargetKind2["equation"] = "equation";
  TargetKind2["subequation"] = "subequation";
  TargetKind2["figure"] = "figure";
  TargetKind2["table"] = "table";
  TargetKind2["code"] = "code";
})(TargetKind || (TargetKind = {}));
var AdmonitionKind;
(function(AdmonitionKind2) {
  AdmonitionKind2["admonition"] = "admonition";
  AdmonitionKind2["attention"] = "attention";
  AdmonitionKind2["caution"] = "caution";
  AdmonitionKind2["danger"] = "danger";
  AdmonitionKind2["error"] = "error";
  AdmonitionKind2["important"] = "important";
  AdmonitionKind2["hint"] = "hint";
  AdmonitionKind2["note"] = "note";
  AdmonitionKind2["seealso"] = "seealso";
  AdmonitionKind2["tip"] = "tip";
  AdmonitionKind2["warning"] = "warning";
})(AdmonitionKind || (AdmonitionKind = {}));

export {
  fileError,
  fileWarn,
  createId,
  normalizeLabel,
  createHtmlId,
  transferTargetAttrs,
  liftChildren,
  setTextAsChild,
  toText,
  copyNode,
  admonitionKindToTitle,
  writeTexLabelledComment,
  getMetadataTags,
  slugToUrl,
  plural,
  remove,
  zwitch,
  matches2 as matches,
  select,
  selectAll,
  PAGE_KNOWN_PARTS,
  SITE_FRONTMATTER_KEYS,
  FRONTMATTER_ALIASES,
  selectBlockParts,
  extractPart,
  parseIndexLine,
  createIndexEntries,
  RuleId,
  isTargetIdentifierNode,
  selectMdastNodes,
  NotebookCell,
  ParseTypesEnum,
  TargetKind,
  AdmonitionKind
};
//# sourceMappingURL=/build/_shared/chunk-Q6DHUCUI.js.map
