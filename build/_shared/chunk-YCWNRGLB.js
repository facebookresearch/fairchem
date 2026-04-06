import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/refractor/lang/hpkp.js
var require_hpkp = __commonJS({
  "../../node_modules/refractor/lang/hpkp.js"(exports, module) {
    module.exports = hpkp;
    hpkp.displayName = "hpkp";
    hpkp.aliases = [];
    function hpkp(Prism) {
      Prism.languages.hpkp = {
        directive: {
          pattern: /\b(?:includeSubDomains|max-age|pin-sha256|preload|report-to|report-uri|strict)(?=[\s;=]|$)/i,
          alias: "property"
        },
        operator: /=/,
        punctuation: /;/
      };
    }
  }
});

export {
  require_hpkp
};
//# sourceMappingURL=/build/_shared/chunk-YCWNRGLB.js.map
