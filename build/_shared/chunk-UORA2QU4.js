import {
  AbstractMermaidTokenBuilder,
  CommonValueConverter,
  EmptyFileSystem,
  MermaidGeneratedSharedModule,
  RadarGeneratedModule,
  __name,
  createDefaultCoreModule,
  createDefaultSharedCoreModule,
  inject,
  lib_exports
} from "/build/_shared/chunk-5TL5RV2T.js";

// ../../node_modules/@mermaid-js/parser/dist/chunks/mermaid-parser.core/chunk-LHMN2FUI.mjs
var RadarTokenBuilder = class extends AbstractMermaidTokenBuilder {
  static {
    __name(this, "RadarTokenBuilder");
  }
  constructor() {
    super(["radar-beta"]);
  }
};
var RadarModule = {
  parser: {
    TokenBuilder: /* @__PURE__ */ __name(() => new RadarTokenBuilder(), "TokenBuilder"),
    ValueConverter: /* @__PURE__ */ __name(() => new CommonValueConverter(), "ValueConverter")
  }
};
function createRadarServices(context = EmptyFileSystem) {
  const shared = inject(
    createDefaultSharedCoreModule(context),
    MermaidGeneratedSharedModule
  );
  const Radar = inject(
    createDefaultCoreModule({ shared }),
    RadarGeneratedModule,
    RadarModule
  );
  shared.ServiceRegistry.register(Radar);
  return { shared, Radar };
}
__name(createRadarServices, "createRadarServices");

export {
  RadarModule,
  createRadarServices
};
//# sourceMappingURL=/build/_shared/chunk-UORA2QU4.js.map
