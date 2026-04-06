import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/refractor/lang/bbcode.js
var require_bbcode = __commonJS({
  "../../node_modules/refractor/lang/bbcode.js"(exports, module) {
    module.exports = bbcode;
    bbcode.displayName = "bbcode";
    bbcode.aliases = ["shortcode"];
    function bbcode(Prism) {
      Prism.languages.bbcode = {
        tag: {
          pattern: /\[\/?[^\s=\]]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'"\]=]+))?(?:\s+[^\s=\]]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'"\]=]+))*\s*\]/,
          inside: {
            tag: {
              pattern: /^\[\/?[^\s=\]]+/,
              inside: {
                punctuation: /^\[\/?/
              }
            },
            "attr-value": {
              pattern: /=\s*(?:"[^"]*"|'[^']*'|[^\s'"\]=]+)/,
              inside: {
                punctuation: [
                  /^=/,
                  {
                    pattern: /^(\s*)["']|["']$/,
                    lookbehind: true
                  }
                ]
              }
            },
            punctuation: /\]/,
            "attr-name": /[^\s=\]]+/
          }
        }
      };
      Prism.languages.shortcode = Prism.languages.bbcode;
    }
  }
});

export {
  require_bbcode
};
//# sourceMappingURL=/build/_shared/chunk-GC3PXAM2.js.map
